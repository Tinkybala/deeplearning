"""Diagnose why RIGID works on some datasets and not others.

Scores several image sets (real and fake, from any dataset) under several
preprocessing pipelines, then reports:
  * mean similarity per set and pipeline
  * AUROC for every real-set x fake-set pair, per pipeline (incl. cross-dataset pairs)
  * a strip plot of the score distributions

Usage:
    uv run python scripts/diagnose.py --config configs/diagnose.yaml

Scores are cached per (set, pipeline) in OUTPUT_DIR/cache, so adding a set or a
pipeline only scores what is new.
"""

from __future__ import annotations

import argparse
import itertools
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rigid import DEFAULT_MEAN, DEFAULT_STD, RIGIDDetector  # noqa: E402
from utils.datasets import IMAGE_EXTS, ImageListDataset, skip_none_collate  # noqa: E402


class JpegRoundtrip:
    """Encode as JPEG at `quality` in memory and decode again (adds identical compression)."""

    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


MATCH_RE = re.compile(r"^match(\d+)(?:_jpeg(\d+))?$")


def min_native_size(name: str, size: int) -> int:
    """Smallest image side a pipeline accepts without upscaling or padding."""
    if name == "native_crop":
        return size
    m = MATCH_RE.match(name)
    return int(m.group(1)) if m else 0


def make_pipeline(name: str, size: int) -> transforms.Compose:
    """Named preprocessing pipelines. All end in ToTensor + ImageNet normalization.

    matchS / matchS_jpegQ: shorter side -> S (antialiased), center crop SxS, optionally a
    JPEG round-trip at quality Q, then resize to size x size. Every set gets the same
    starting size, the same final resize factor and (with jpeg) the same compression.
    """
    norm = [transforms.ToTensor(), transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)]
    m = MATCH_RE.match(name)
    if m:
        s = int(m.group(1))
        pre = [transforms.Resize(s), transforms.CenterCrop(s)]
        if m.group(2):
            pre.append(JpegRoundtrip(int(m.group(2))))
        pre.append(transforms.Resize((size, size)))
    elif name == "squash":            # paper / notebook: resize to size x size, ignoring aspect
        pre = [transforms.Resize((size, size))]
    elif name == "short_crop":      # keep aspect: shorter side -> size*256/224, center crop
        pre = [transforms.Resize(round(size * 256 / 224)), transforms.CenterCrop(size)]
    elif name == "native_crop":     # no resampling at all: center crop at native resolution
        pre = [transforms.CenterCrop(size)]
    else:
        raise ValueError(f"Unknown pipeline {name!r}")
    return transforms.Compose(pre + norm)


def list_images(folder: Path, n: int, seed: int) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    return sorted(random.Random(seed).sample(files, min(n, len(files))))


def score_set(detector, files, transform, cfg) -> np.ndarray:
    ds = ImageListDataset([(p, 0, "") for p in files], transform)
    loader = DataLoader(ds, batch_size=cfg["BATCH_SIZE"], num_workers=cfg["NUM_WORKERS"],
                        collate_fn=skip_none_collate)
    gen = torch.Generator().manual_seed(cfg["SEED"])
    out = []
    for batch in loader:
        if batch is not None:
            out.append(detector.score_batch(batch[0], n_noise=cfg["N_NOISE"], generator=gen))
    return torch.cat(out).numpy()


def plot(df: pd.DataFrame, sets: dict, pipelines: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"real": "#2a78d6", "fake": "#eb6834"}   # categorical slots 1 and 2
    names = list(sets)
    ncol = min(4, len(pipelines))
    nrow = -(-len(pipelines) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, (0.42 * len(names) + 1.2) * nrow + 0.4),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax in axes.flat[len(pipelines):]:
        ax.set_visible(False)
    for ax, pipe in zip(axes.flat, pipelines):
        for i, name in enumerate(names):
            s = df[(df.set == name) & (df.pipeline == pipe)].similarity.to_numpy()
            if len(s) == 0:
                continue
            kind = sets[name]["label"]
            y = i + rng.uniform(-0.18, 0.18, len(s))
            ax.scatter(s, y, s=6, alpha=0.35, color=colors[kind], linewidths=0)
            ax.plot([np.median(s)] * 2, [i - 0.32, i + 0.32], color="#0b0b0b", lw=2)
        ax.set_title(pipe, fontsize=11)
        ax.set_xlabel("cosine similarity (clean vs noisy)")
        ax.grid(axis="x", color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for row in axes:
        row[0].set_yticks(range(len(names)), names)
    axes[0][0].invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
    handles.append(plt.Line2D([], [], color="#0b0b0b", lw=2, label="median"))
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False)
    fig.suptitle("RIGID similarity by image set and preprocessing", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.05 / nrow))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = {"N_NOISE": 1, "LAMB": 0.05, "MODEL_NAME": "dinov2_vitl14", "IMAGE_SIZE": 224,
           "IMAGES_PER_SET": 300, "BATCH_SIZE": 32, "NUM_WORKERS": 4, "DEVICE": None,
           "SEED": 100, "OUTPUT_DIR": "results/diagnose",
           **yaml.safe_load(args.config.read_text())}
    sets: dict = cfg["SETS"]
    pipelines: list[str] = cfg["PIPELINES"]

    out_dir = ROOT / cfg["OUTPUT_DIR"]
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    detector = RIGIDDetector(lamb=cfg["LAMB"], model_name=cfg["MODEL_NAME"], device=cfg["DEVICE"],
                             image_size=cfg["IMAGE_SIZE"], seed=cfg["SEED"])
    tag = f"{cfg['MODEL_NAME']}_l{cfg['LAMB']}_s{cfg['IMAGE_SIZE']}_n{cfg['IMAGES_PER_SET']}_k{cfg['N_NOISE']}"

    rows = []
    jobs = list(itertools.product(sets, pipelines))
    for name, pipe in tqdm(jobs, desc="Sets x pipelines"):
        spec = sets[name]
        # Skip sets that would need upscaling / padding for this pipeline.
        if spec.get("native_size", 10**9) < min_native_size(pipe, cfg["IMAGE_SIZE"]):
            continue
        f = cache / f"{name}__{pipe}__{tag}.json"
        if f.exists():
            rec = json.loads(f.read_text())
        else:
            files = list_images(Path(spec["path"]).expanduser(), cfg["IMAGES_PER_SET"], cfg["SEED"])
            scores = score_set(detector, files, make_pipeline(pipe, cfg["IMAGE_SIZE"]), cfg)
            rec = {"scores": scores.tolist()}
            f.write_text(json.dumps(rec))
        rows += [{"set": name, "label": spec["label"], "pipeline": pipe, "similarity": s} for s in rec["scores"]]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scores.csv", index=False)

    # Mean similarity per set
    means = df.pivot_table(index="set", columns="pipeline", values="similarity", aggfunc="mean")
    means = means.reindex(index=list(sets), columns=pipelines)

    # AUROC per real x fake pair per pipeline
    reals = [s for s in sets if sets[s]["label"] == "real"]
    fakes = [s for s in sets if sets[s]["label"] == "fake"]
    auc_rows = []
    for pipe in pipelines:
        for r, fk in itertools.product(reals, fakes):
            a = df[(df.set == r) & (df.pipeline == pipe)].similarity.to_numpy()
            b = df[(df.set == fk) & (df.pipeline == pipe)].similarity.to_numpy()
            if len(a) and len(b):
                y = np.r_[np.ones(len(a)), np.zeros(len(b))]
                auc_rows.append({"pipeline": pipe, "real": r, "fake": fk,
                                 "auroc": roc_auc_score(y, np.r_[a, b])})
    aucs = pd.DataFrame(auc_rows)
    aucs.to_csv(out_dir / "auroc_pairs.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50, "display.precision", 4)
    print("\nMean similarity per set (higher = more 'real'-looking to RIGID):")
    print(means.round(4).to_string())
    for pipe in pipelines:
        sub = aucs[aucs.pipeline == pipe]
        if len(sub):
            print(f"\nAUROC, pipeline = {pipe}  (rows = real set, cols = fake set; 0.5 = chance, <0.5 = inverted)")
            print(sub.pivot(index="real", columns="fake", values="auroc")
                  .reindex(index=reals, columns=fakes).round(3).to_string())

    # Summary: mean AUROC over named groups of pairs (from config SUMMARY), per pipeline.
    if cfg.get("SUMMARY") and len(aucs):
        summ = {}
        for label, pair in cfg["SUMMARY"].items():
            sub = aucs[(aucs.real == pair["real"]) & (aucs.fake.isin(pair["fakes"]))]
            summ[label] = sub.groupby("pipeline").auroc.mean().reindex(pipelines)
        print("\nSummary: mean AUROC per pair group (NaN = sets skipped for that pipeline)")
        print(pd.DataFrame(summ).T.round(3).to_string())

    plot(df, sets, pipelines, out_dir / "similarity_strip.png")
    print(f"\nSaved scores.csv, auroc_pairs.csv and similarity_strip.png to {out_dir}")


if __name__ == "__main__":
    main()
