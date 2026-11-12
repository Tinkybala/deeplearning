"""Evaluate RIGID on a registered dataset.

Usage:
    uv run python scripts/evaluate.py --config configs/cifake.yaml

Writes results/<config name>/scores.csv (one row per image) and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rigid import RIGIDDetector, calibrate_threshold
from utils import load_dataset, skip_none_collate

REQUIRED = ["THRESHOLD", "N_NOISE", "LAMB", "MODEL_NAME", "DATASET_NAME"]
DEFAULTS = {
    "SPLIT": None,
    "MAX_IMAGES_PER_CLASS": None,
    "DATA_ROOT": None,
    "REAL_DATA_ROOT": None,
    "FAKE_DATA_ROOT": None,
    "DOWNLOAD_DIR": None,
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 2,
    "DEVICE": None,
    "SEED": 100,
    "OUTPUT_DIR": "results",
}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"Config {path} is missing: {', '.join(missing)}")
    return {**DEFAULTS, **cfg}


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """labels: 1 = real, 0 = AI-generated. Higher score = more likely real."""
    # Threshold-free metrics, same convention as the IBM notebook (real = positive class).
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    fpr95 = float(fpr[np.argmax(tpr >= 0.95)])

    # Thresholded metrics, with AI-generated as the positive (detected) class.
    pred_fake = scores < threshold
    is_fake = labels == 0
    tp = int((pred_fake & is_fake).sum())
    fp = int((pred_fake & ~is_fake).sum())
    fn = int((~pred_fake & is_fake).sum())
    tn = int((~pred_fake & ~is_fake).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    best_t = calibrate_threshold(scores[labels == 1], scores[labels == 0])
    best_acc = 0.5 * ((scores[labels == 1] >= best_t).mean() + (scores[labels == 0] < best_t).mean())

    return {
        "auroc": float(metrics.roc_auc_score(labels, scores)),
        "average_precision": float(metrics.average_precision_score(labels, scores)),
        "fpr_at_95_tpr": fpr95,
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "real_accuracy": tn / (tn + fp) if tn + fp else 0.0,
        "fake_accuracy": recall,
        "fake_precision": precision,
        "fake_recall": recall,
        "fake_f1": f1,
        "confusion": {"fake_as_fake": tp, "real_as_fake": fp, "fake_as_real": fn, "real_as_real": tn},
        "best_threshold": best_t,
        "best_threshold_balanced_accuracy": float(best_acc),
        "mean_score_real": float(scores[labels == 1].mean()),
        "mean_score_fake": float(scores[labels == 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    # Resolve all configured relative paths against the project directory rather
    # than depending on the shell's current working directory.
    for key in ("DATA_ROOT", "REAL_DATA_ROOT", "FAKE_DATA_ROOT", "DOWNLOAD_DIR"):
        value = cfg.get(key)
        if value:
            path = Path(value).expanduser()
            cfg[key] = str(path if path.is_absolute() else (ROOT / path).resolve())

    torch.manual_seed(cfg["SEED"])
    detector = RIGIDDetector(
        lamb=cfg["LAMB"], model_name=cfg["MODEL_NAME"], device=cfg["DEVICE"],
        threshold=cfg["THRESHOLD"], seed=cfg["SEED"],
    )
    dataset = load_dataset(
        cfg["DATASET_NAME"], split=cfg["SPLIT"], transform=detector.transform,
        max_per_class=cfg["MAX_IMAGES_PER_CLASS"], data_root=cfg["DATA_ROOT"],
        real_data_root=cfg["REAL_DATA_ROOT"], fake_data_root=cfg["FAKE_DATA_ROOT"],
        cache_dir=cfg["DOWNLOAD_DIR"], seed=cfg["SEED"],
    )
    n_real = sum(dataset.labels)
    print(f"Dataset {cfg['DATASET_NAME']}: {n_real} real + {len(dataset) - n_real} fake | "
          f"model {cfg['MODEL_NAME']} on {detector.device}")

    loader = DataLoader(dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                        num_workers=cfg["NUM_WORKERS"], collate_fn=skip_none_collate)
    gen = torch.Generator().manual_seed(cfg["SEED"])

    scores, labels, paths = [], [], []
    start = time.time()
    for batch in tqdm(loader, desc="Scoring"):
        if batch is None:
            continue
        x, y, p = batch
        scores.append(detector.score_batch(x, n_noise=cfg["N_NOISE"], generator=gen))
        labels.append(y)
        paths.extend(p)
    elapsed = time.time() - start
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    scored = set(paths)
    skipped = [str(p) for p, _, _ in dataset.items if str(p) not in scored]
    if skipped:
        print(f"Skipped {len(skipped)} unreadable image(s): {skipped[:5]}")

    results = compute_metrics(scores, labels, cfg["THRESHOLD"])

    # Per-generator results: each fake group against all real images (like the IBM notebook).
    group_of = {str(p): g for p, _, g in dataset.items}
    groups = np.array([group_of[p] for p in paths])
    fake_groups = sorted(set(groups[labels == 0]))
    per_group = {}
    if len(fake_groups) > 1:
        real_mask = labels == 1
        for g in fake_groups:
            mask = real_mask | (groups == g)
            per_group[g] = compute_metrics(scores[mask], labels[mask], cfg["THRESHOLD"])

    out_dir = ROOT / cfg["OUTPUT_DIR"] / args.config.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "similarity", "predicted"])
        for path, y, g, s in zip(paths, labels, groups, scores):
            source_path = Path(path)
            try:
                portable_path = source_path.relative_to(ROOT).as_posix()
            except ValueError:
                portable_path = source_path.name
            w.writerow([portable_path, "real" if y == 1 else "fake", g, f"{s:.6f}",
                        "real" if s >= cfg["THRESHOLD"] else "fake"])
    public_cfg = dict(cfg)
    for key in ("DATA_ROOT", "REAL_DATA_ROOT", "FAKE_DATA_ROOT", "DOWNLOAD_DIR"):
        if public_cfg.get(key):
            path = Path(public_cfg[key])
            try:
                public_cfg[key] = path.relative_to(ROOT).as_posix()
            except ValueError:
                public_cfg[key] = path.name
    portable_skipped = []
    for skipped_path in skipped:
        path = Path(skipped_path)
        try:
            portable_skipped.append(path.relative_to(ROOT).as_posix())
        except ValueError:
            portable_skipped.append(path.name)
    summary = {"config": public_cfg, "num_images": len(scores), "seconds": round(elapsed, 1),
               "skipped_images": portable_skipped, "metrics": results, "per_group": per_group}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if per_group:
        print(f"\n{'fake group':<24} {'AUROC':>7} {'AP':>7} {'FPR95':>7} {'fake acc':>9} {'mean sim':>9}")
        for g, r in per_group.items():
            print(f"{g:<24} {r['auroc']:>7.4f} {r['average_precision']:>7.4f} {r['fpr_at_95_tpr']:>7.4f} "
                  f"{r['fake_accuracy']:>9.4f} {r['mean_score_fake']:>9.4f}")
        print("-" * 68)
        print(f"{'mean over groups':<24} {np.mean([r['auroc'] for r in per_group.values()]):>7.4f} "
              f"{np.mean([r['average_precision'] for r in per_group.values()]):>7.4f} "
              f"{np.mean([r['fpr_at_95_tpr'] for r in per_group.values()]):>7.4f}")

    m = results
    print(f"\nOverall: AUROC {m['auroc']:.4f} | AP {m['average_precision']:.4f} | FPR@95TPR {m['fpr_at_95_tpr']:.4f}")
    print(f"Mean similarity: real {m['mean_score_real']:.4f} | fake {m['mean_score_fake']:.4f}")
    print(f"At THRESHOLD={m['threshold']}: accuracy {m['accuracy']:.4f} "
          f"(real {m['real_accuracy']:.4f}, fake {m['fake_accuracy']:.4f}) | fake F1 {m['fake_f1']:.4f}")
    print(f"Best threshold on this data: {m['best_threshold']:.4f} "
          f"(balanced accuracy {m['best_threshold_balanced_accuracy']:.4f})")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
