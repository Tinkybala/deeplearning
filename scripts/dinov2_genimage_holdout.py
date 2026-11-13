"""Frozen-DINOv2 detectors on the same GenImage generator holdout as the ResNet study.

This puts Michael's RIGID / DINOv2-probe detectors on Tyrone's ResNet-18 protocol so
the two families can be compared in one table. The split, test-time corruptions,
threshold rule and metrics are imported from `train_resnet_generalization.py`, so they
are identical by construction:

- Fit: generated train images of the six seen generators + 80% of the unique real
  train photos.
- Seen validation: generated val images of the seen generators + the reserved 20% of
  real train photos. Used only to pick each detector's threshold.
- Test: the held-out generator's 500 real + 500 generated images, under the same six
  conditions (clean, JPEG Q50, resize 128/64/32, resize 128 + JPEG Q50).

Detectors (all use one frozen DINOv2 backbone; nothing in it is trained):
    rigid              plain RIGID cosine similarity between clean and noisy embeddings
    logreg_clean       logistic regression on the clean embedding (UnivFD-style control)
    logreg_clean+diff  logistic regression on clean + (noisy - clean)  [Michael's probe]
    mlp_clean+diff     the MLP version of that probe

Embeddings are cached per image folder under results/dinov2_embeddings/ (git-ignored,
keyed by file name so the cache is portable). Rotating the held-out generator therefore
only embeds folders that no earlier run needed.

    python scripts/dinov2_genimage_holdout.py --held-out imagenet_glide
    python scripts/dinov2_genimage_holdout.py --held-out imagenet_glide --augmentation jpeg
    python scripts/dinov2_genimage_holdout.py --held-out all          # leave-one-generator-out
    python scripts/compare_holdout_results.py --held-out imagenet_glide
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_resnet_generalization as resnet  # noqa: E402  shared split, conditions, metrics
from rigid import DEFAULT_MEAN, DEFAULT_STD, DINOV2_MODELS, RIGIDDetector  # noqa: E402
from rigid_probe import build_features  # noqa: E402
from train_final_probe import train_mlp  # noqa: E402

CONDITIONS = ["clean", "jpeg_q50", "resize_128", "resize_64", "resize_32", "resize_jpeg_128"]
DETECTORS = ["rigid", "logreg_clean", "logreg_clean+diff", "mlp_clean+diff"]
EMBED_ROOT = ROOT / "results" / "dinov2_embeddings"
PROBE_ROOT = ROOT / "model" / "DINOv2" / "Frozen DINO model" / "checkpoints"


# ---------------------------------------------------------------------------- data

def build_transform(preprocess: str, augmentation: str) -> transforms.Compose:
    """Eval preprocessing, optionally preceded by the ResNet study's random artifacts.

    crop   = ResNet eval pipeline (shorter side 256, center crop 224); default, so both
             model families see the same pixels.
    squash = RIGID paper / Michael's runs (resize to 224x224, aspect ratio ignored).
    """
    artifacts = []
    if augmentation != "none":
        artifacts = [resnet.RandomImageArtifacts(use_jpeg=augmentation in ("jpeg", "both"),
                                                 use_resize=augmentation in ("resize", "both"))]
    tail = [transforms.ToTensor(), transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)]
    if preprocess == "crop":
        return transforms.Compose([transforms.Resize(256), *artifacts, transforms.CenterCrop(224), *tail])
    return transforms.Compose([*artifacts, transforms.Resize((224, 224)), *tail])


def subsample(records: list, per_group: int | None, seed: int) -> list:
    """Keep at most `per_group` records per (generator, label); for quick smoke runs."""
    if per_group is None:
        return records
    groups = defaultdict(list)
    for rec in records:
        groups[(rec[2], rec[1])].append(rec)
    rng = random.Random(seed)
    kept = []
    for key in sorted(groups):
        items = groups[key]
        kept.extend(items if len(items) <= per_group else rng.sample(items, per_group))
    return kept


def portable(path, test_root: Path) -> str:
    """Project- or dataset-relative path, so outputs and caches carry no local paths."""
    p = Path(path)
    for base in (ROOT, test_root.parent):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            pass
    return p.name


class EmbeddingStore:
    """Clean/noisy DINOv2 embeddings, cached per (image folder, test condition, augmentation).

    Each cache file holds the file names it covers; missing images are embedded and
    appended, so smoke runs and later holdouts extend the same cache. Noise and artifact
    augmentation are seeded from the folder name and the number of images already cached.
    """

    def __init__(self, args):
        self.args = args
        tag = f"_max{args.max_per_group}" if args.max_per_group else ""
        self.root = EMBED_ROOT / f"{args.model_name}_{args.preprocess}_lamb{args.lamb}_seed{args.seed}{tag}"
        self.detector = None
        self.memo: dict = {}
        self.embedded = 0

    def model(self) -> RIGIDDetector:
        if self.detector is None:
            self.detector = RIGIDDetector(lamb=self.args.lamb, model_name=self.args.model_name,
                                          device=self.args.device, seed=self.args.seed)
        return self.detector

    @torch.no_grad()
    def _embed(self, files: list[Path], condition: str, augmentation: str, seed: int, desc: str) -> dict:
        det, args = self.model(), self.args
        random.seed(seed)  # artifact augmentation when workers == 0
        dataset = resnet.ImageRecords([(p, 0, "") for p in files],
                                      build_transform(args.preprocess, augmentation), condition)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(seed))
        noise_gen = torch.Generator().manual_seed(seed)
        clean, noisy = [], []
        for x, _, _, _ in tqdm(loader, desc=desc, leave=False):
            x = x.to(det.device)
            noise = torch.randn(x.shape, generator=noise_gen).to(det.device)
            clean.append(det.model(x).float().cpu())
            noisy.append(det.model(x + noise * det.lamb).float().cpu())
        self.embedded += len(files)
        return {"clean": torch.cat(clean).half().numpy(), "noisy": torch.cat(noisy).half().numpy(),
                "name": np.asarray([p.name for p in files])}

    def folder(self, folder: Path, names: set[str], condition: str, augmentation: str) -> dict:
        rel = portable(folder, self.args.test_data_root)
        key = (rel, condition, augmentation)
        data = self.memo.get(key)
        cache = self.root / rel / (condition + ("" if augmentation == "none" else f"__aug-{augmentation}") + ".npz")
        if data is None and cache.exists():
            try:
                data = dict(np.load(cache))
            except (EOFError, OSError, ValueError, zipfile.BadZipFile) as err:
                # e.g. a file that iCloud "Optimize Mac Storage" evicted and failed to restore
                print(f"Unreadable cache {cache.relative_to(ROOT)} ({err!r}); re-embedding this folder")
                data = None
        missing = sorted(names - set(data["name"])) if data is not None else sorted(names)
        if missing:
            seed = (self.args.seed * 1_000_003 + zlib.crc32(rel.encode())
                    + (0 if data is None else len(data["name"]))) % 2**31
            label = f"{rel} [{condition}{'' if augmentation == 'none' else ', aug ' + augmentation}]"
            new = self._embed([folder / n for n in missing], condition, augmentation, seed, label)
            data = new if data is None else {k: np.concatenate([data[k], new[k]]) for k in ("clean", "noisy", "name")}
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **{k: data[k] for k in ("clean", "noisy", "name")})
        data["index"] = {n: i for i, n in enumerate(data["name"])}
        self.memo[key] = data
        return data

    def gather(self, records: list, condition: str, augmentation: str) -> dict:
        """Embeddings for `records` (path, label, generator), in record order."""
        by_folder = defaultdict(list)
        for i, (path, _, _) in enumerate(records):
            by_folder[Path(path).parent].append(i)
        clean = noisy = None
        for folder, idxs in by_folder.items():
            names = {Path(records[i][0]).name for i in idxs}
            data = self.folder(folder, names, condition, augmentation)
            rows = [data["index"][Path(records[i][0]).name] for i in idxs]
            if clean is None:
                dim = data["clean"].shape[1]
                clean = np.empty((len(records), dim), np.float16)
                noisy = np.empty((len(records), dim), np.float16)
            clean[idxs], noisy[idxs] = data["clean"][rows], data["noisy"][rows]
        return {"clean": clean, "noisy": noisy,
                "label": np.asarray([r[1] for r in records], dtype=np.int8),
                "group": np.asarray([r[2] for r in records]),
                "path": np.asarray([str(r[0]) for r in records])}


# ---------------------------------------------------------------------- detectors

def fit_detector(name: str, fit: dict, args, held_out: str):
    """Return (score_fn, bundle); score_fn maps embeddings to a score, higher = more real."""
    if name == "rigid":
        return (lambda e: build_features(e["clean"], e["noisy"], "cos")[:, 0]), None
    model, features = name.split("_", 1)
    X = build_features(fit["clean"], fit["noisy"], features)
    y = fit["label"].astype(np.int64)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    if model == "logreg":
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=5000).fit(Xs, y)
        real_col = list(clf.classes_).index(1)

        def score(e):
            return clf.predict_proba(scaler.transform(build_features(e["clean"], e["noisy"], features)))[:, real_col]
        bundle = {"scaler": scaler, "clf": clf}
    else:
        net = train_mlp(Xs, y, args.hidden, seed=args.seed)

        def score(e):
            Xe = torch.tensor(scaler.transform(build_features(e["clean"], e["noisy"], features)), dtype=torch.float32)
            with torch.no_grad():
                return torch.sigmoid(net(Xe)).squeeze(1).numpy()
        bundle = {"scaler": scaler, "state_dict": net.state_dict(), "in_dim": Xs.shape[1], "hidden": args.hidden}
    bundle["meta"] = {"model": model, "features": features, "lamb": args.lamb, "model_name": args.model_name,
                      "preprocess": args.preprocess, "seed": args.seed, "held_out": held_out}
    return score, bundle


DESCRIPTIONS = {
    "rigid": "RIGID cosine similarity (frozen DINOv2, no training); score is the similarity, not a probability",
    "logreg_clean": "Logistic regression on frozen DINOv2 clean CLS embedding",
    "logreg_clean+diff": "Logistic regression on frozen DINOv2 clean embedding + (noisy - clean)",
    "mlp_clean+diff": "MLP probe on frozen DINOv2 clean embedding + (noisy - clean)",
}
AUGMENTATION_TEXT = {
    "none": "none (frozen embeddings of center-cropped images)",
    "jpeg": "random JPEG quality 30-95 with p=0.5, applied once before embedding",
    "resize": "random downsampling to 32/64/128 px with p=0.5, applied once before embedding",
    "both": "random JPEG 30-95 and downsampling to 32/64/128 px, each p=0.5, applied once before embedding",
}


def run_name(det: str, args, held_out: str) -> str:
    parts = ["dinov2"]
    if args.model_name != "dinov2_vitl14":
        parts.append(args.model_name.removeprefix("dinov2_"))
    parts.append(det.replace("+", "_"))
    if args.augmentation != "none":
        parts.append(args.augmentation)
    if args.preprocess != "crop":
        parts.append(args.preprocess)
    if args.max_per_group:
        parts.append(f"max{args.max_per_group}")
    return "_".join(parts) + f"_holdout_{held_out}"


# --------------------------------------------------------------------------- main

def run_holdout(held_out: str, store: EmbeddingStore, args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- split: same code path and seed as train_resnet_generalization.main ----
    seen = [g for g in resnet.GENERATORS if g != held_out]
    real_train_paths = resnet.files_for(args.data_root, seen[0], "train", "nature")
    random.Random(args.seed).shuffle(real_train_paths)
    calibration_count = max(1, len(real_train_paths) // 5)
    fit_records = resnet.records_for(args.data_root, seen, "train", real_paths=real_train_paths[calibration_count:])
    val_records = resnet.records_for(args.data_root, seen, "val", real_paths=real_train_paths[:calibration_count])
    test_records = resnet.records_for(args.test_data_root, [held_out], args.test_split)
    fit_records = subsample(fit_records, args.max_per_group, args.seed)
    val_records = subsample(val_records, args.max_per_group, args.seed)
    test_records = subsample(test_records, args.max_per_group, args.seed)
    print(f"\n=== Held out {held_out}: fit {len(fit_records)}, seen-val {len(val_records)}, test {len(test_records)}")

    start, before = time.time(), store.embedded
    fit = store.gather(fit_records, "clean", args.augmentation)
    val = store.gather(val_records, "clean", "none")
    tests = {c: store.gather(test_records, c, "none") for c in CONDITIONS}
    embed_seconds, newly_embedded = round(time.time() - start, 1), store.embedded - before
    device_name = str(store.detector.device) if store.detector is not None else "cached"

    for det in args.detectors:
        if det == "rigid" and args.augmentation != "none":
            print("rigid: skipped (it uses no fit images, so augmentation cannot change it)")
            continue
        t0 = time.time()
        score_fn, bundle = fit_detector(det, fit, args, held_out)
        val_scores = score_fn(val)
        threshold, val_ba = resnet.best_threshold(val["label"].astype(int), val_scores)
        metrics, rows = {}, []
        for cond, emb in tests.items():
            scores = score_fn(emb)
            labels = emb["label"].astype(int)
            metrics[cond] = resnet.summarize(labels, scores, threshold)
            rows.extend((cond, g, portable(p, args.test_data_root), "real" if y == 1 else "fake",
                         f"{s:.7f}", "real" if s >= threshold else "fake")
                        for g, p, y, s in zip(emb["group"], emb["path"], labels, scores))

        name = run_name(det, args, held_out)
        out_dir = ROOT / "results" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "test_scores.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "generator", "path", "label", "p_real", "prediction"])
            writer.writerows(rows)
        if bundle is not None:
            probe_path = PROBE_ROOT / name / "probe.joblib"
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, probe_path)
        summary = {
            "model": DESCRIPTIONS[det],
            "backbone": args.model_name,
            "detector": det,
            "preprocess": args.preprocess,
            "rigid_noise_lambda": args.lamb,
            "training_data_root": resnet.portable_root(args.data_root),
            "test_data_root": resnet.portable_root(args.test_data_root),
            "test_split": args.test_split,
            "held_out_generator": held_out,
            "seen_generators": seen,
            "train_images": len(fit_records),
            "validation_images": len(val_records),
            "test_images": len(test_records),
            "max_images_per_group": args.max_per_group,
            "training_augmentation": AUGMENTATION_TEXT[args.augmentation],
            "threshold_selected_on_seen_validation": threshold,
            "seen_validation_balanced_accuracy": val_ba,
            "seen_validation_auroc": float(roc_auc_score(val["label"], val_scores)),
            "probe_hyperparameters": None if det == "rigid" else (
                {"C": args.C} if det.startswith("logreg") else {"hidden": args.hidden}),
            "seed": args.seed,
            "device": device_name,
            "images_embedded_this_run": newly_embedded,
            "embedding_seconds_this_run": embed_seconds,
            "probe_and_evaluation_seconds": round(time.time() - t0, 1),
            "torch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
            "test_metrics": metrics,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{det}: seen-val BA {val_ba:.3f}, threshold {threshold:.5f} -> results/{name}")
        for cond in CONDITIONS:
            m = metrics[cond]
            print(f"  {cond:16s} BA {m['balanced_accuracy']:.3f}  AUROC {m['auroc_real_positive']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=resnet.DEFAULT_DATA_ROOT)
    parser.add_argument("--test-data-root", type=Path, default=resnet.DEFAULT_TEST_DATA_ROOT)
    parser.add_argument("--test-split", default="test",
                        help="split folder under each held-out generator in --test-data-root")
    parser.add_argument("--held-out", nargs="+", default=["imagenet_glide"],
                        choices=[*resnet.GENERATORS, "all"],
                        help="one or more held-out generators, or 'all' for leave-one-generator-out")
    parser.add_argument("--detectors", nargs="+", default=DETECTORS, choices=DETECTORS)
    parser.add_argument("--augmentation", choices=("none", "jpeg", "resize", "both"), default="none",
                        help="artifact augmentation applied to fit images before embedding (as in the ResNet ablation)")
    parser.add_argument("--preprocess", choices=("crop", "squash"), default="crop")
    parser.add_argument("--model-name", default="dinov2_vitl14", choices=DINOV2_MODELS)
    parser.add_argument("--lamb", type=float, default=0.05, help="RIGID noise strength")
    parser.add_argument("--C", type=float, default=0.1, help="logreg inverse L2 strength")
    parser.add_argument("--hidden", type=int, default=256, help="MLP hidden units")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="split seed; 42 matches the ResNet runs")
    parser.add_argument("--max-per-group", type=int, default=None,
                        help="cap images per (generator, label) in every split; for smoke tests only")
    args = parser.parse_args()
    for key in ("data_root", "test_data_root"):
        value = getattr(args, key).expanduser()
        setattr(args, key, (value if value.is_absolute() else ROOT / value).resolve())

    held_outs = resnet.GENERATORS if "all" in args.held_out else list(dict.fromkeys(args.held_out))
    store = EmbeddingStore(args)
    for held_out in held_outs:
        run_holdout(held_out, store, args)
    print(f"\nEmbedded {store.embedded} new images. Compare runs with: "
          f"python scripts/compare_holdout_results.py --held-out {held_outs[0]}")


if __name__ == "__main__":
    main()
