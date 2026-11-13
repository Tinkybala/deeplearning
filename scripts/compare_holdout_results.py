"""Tabulate every results/*_holdout_<generator>/ run side by side, with bootstrap CIs.

Works for the ResNet runs (train_resnet_generalization.py) and the frozen-DINOv2 runs
(dinov2_genimage_holdout.py), since both write the same summary.json / test_scores.csv
format. Writes results/comparison_holdout_<generator>.md with:

1. Balanced accuracy and AUROC per test condition, each with a 95% bootstrap interval.
2. Paired differences against a reference run (default: the plain ResNet). Every run is
   scored on the same test images, so each bootstrap replicate resamples the same images
   for all runs; the difference interval therefore reflects only how the two models
   disagree, not the shared test-set sampling noise. `*` marks intervals excluding zero.

The bootstrap is stratified (real and generated images are resampled separately, so each
replicate keeps 500 + 500) and uses each run's saved, validation-calibrated predictions;
thresholds are never re-fitted on test data.

    python scripts/compare_holdout_results.py --held-out imagenet_glide
    python scripts/compare_holdout_results.py --held-out imagenet_glide --reference dinov2_logreg_clean
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [
    ("clean", "Clean"),
    ("jpeg_q50", "JPEG Q50"),
    ("resize_128", "Resize 128"),
    ("resize_64", "Resize 64"),
    ("resize_32", "Resize 32"),
    ("resize_jpeg_128", "128 + JPEG 50"),
]
METRICS = [("balanced_accuracy", "Balanced accuracy"), ("auroc_real_positive", "AUROC (real positive)")]


def load_runs(held_out: str, include_smoke: bool) -> dict:
    suffix = f"_holdout_{held_out}"
    runs = {}
    for path in sorted((ROOT / "results").glob(f"*{suffix}/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("max_images_per_group") and not include_smoke:
            continue
        if not all(key in summary.get("test_metrics", {}) for key, _ in CONDITIONS):
            continue
        runs[path.parent.name.removesuffix(suffix)] = {"summary": summary, "dir": path.parent}
    return runs


def load_scores(run_dir: Path) -> dict:
    """condition -> dict of path-sorted arrays (path, is_real, score, predicted_real)."""
    with open(run_dir / "test_scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for cond, _ in CONDITIONS:
        rs = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["path"])
        out[cond] = {"path": np.array([r["path"] for r in rs]),
                     "real": np.array([r["label"] == "real" for r in rs]),
                     "score": np.array([float(r["p_real"]) for r in rs]),
                     "pred_real": np.array([r["prediction"] == "real" for r in rs])}
    return out


def bootstrap_metrics(data: dict, idx: np.ndarray, n_real: int) -> dict:
    """Balanced accuracy and AUROC for every bootstrap replicate (rows of idx).

    idx[:, :n_real] index real images and idx[:, n_real:] generated images.
    """
    pred = data["pred_real"][idx]
    ba = 0.5 * (pred[:, :n_real].mean(1) + (~pred[:, n_real:]).mean(1))
    ranks = rankdata(data["score"][idx], axis=1)          # ties get average ranks
    n_fake = idx.shape[1] - n_real
    auc = (ranks[:, :n_real].sum(1) - n_real * (n_real + 1) / 2) / (n_real * n_fake)
    return {"balanced_accuracy": ba, "auroc_real_positive": auc}


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out", default="imagenet_glide")
    parser.add_argument("--reference", default="resnet", help="run name for the paired-difference table")
    parser.add_argument("--bootstrap", type=int, default=2000, help="replicates; 0 = point estimates only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-smoke", action="store_true", help="also list runs made with --max-per-group")
    args = parser.parse_args()

    runs = load_runs(args.held_out, args.include_smoke)
    if not runs:
        raise SystemExit(f"No complete results found for results/*_holdout_{args.held_out}/")

    # ---- paired, stratified bootstrap over the shared test images ----
    boot = {}
    if args.bootstrap:
        scores = {name: load_scores(run["dir"]) for name, run in runs.items()}
        first = next(iter(scores.values()))
        for name, sc in scores.items():
            for cond, _ in CONDITIONS:
                if not np.array_equal(sc[cond]["path"], first[cond]["path"]):
                    raise SystemExit(f"{name} was scored on different {cond} images; cannot pair runs")
        rng = np.random.default_rng(args.seed)
        for cond, _ in CONDITIONS:
            is_real = first[cond]["real"]
            real_idx, fake_idx = np.flatnonzero(is_real), np.flatnonzero(~is_real)
            idx = np.concatenate([rng.choice(real_idx, (args.bootstrap, len(real_idx))),
                                  rng.choice(fake_idx, (args.bootstrap, len(fake_idx)))], axis=1)
            for name in runs:
                boot[(name, cond)] = bootstrap_metrics(scores[name][cond], idx, len(real_idx))

    names = list(runs)
    fmt = {"balanced_accuracy": lambda v: f"{v:.1%}", "auroc_real_positive": lambda v: f"{v:.3f}"}
    header = "| Run | Seen-val BA | " + " | ".join(label for _, label in CONDITIONS) + " |"
    rule = "|---|---:|" + "---:|" * len(CONDITIONS)
    sections = [f"# Held-out `{args.held_out}` comparison",
                f"Test images per condition: {', '.join(sorted({str(r['summary']['test_images']) for r in runs.values()}))}. "
                "Each run's threshold was calibrated on its own seen-generator validation split. "
                + (f"Brackets: 95% stratified bootstrap interval ({args.bootstrap} replicates, paired across runs)."
                   if args.bootstrap else "")]

    for metric, title in METRICS:
        best = {c: max(runs[n]["summary"]["test_metrics"][c][metric] for n in names) for c, _ in CONDITIONS}
        lines = [header, rule]
        for name in names:
            cells = []
            for cond, _ in CONDITIONS:
                value = runs[name]["summary"]["test_metrics"][cond][metric]
                text = fmt[metric](value)
                if len(names) > 1 and value == best[cond]:
                    text = f"**{text}**"
                if boot:
                    lo, hi = ci(boot[(name, cond)][metric])
                    text += f" [{fmt[metric](lo)}, {fmt[metric](hi)}]".replace("%", "")
                cells.append(text)
            val = runs[name]["summary"].get("seen_validation_balanced_accuracy")
            lines.append(f"| {name} | {'' if val is None else f'{val:.1%}'} | " + " | ".join(cells) + " |")
        sections += [f"## {title}", "\n".join(lines)]

    if boot and args.reference in runs:
        for metric, title in METRICS:
            scale, unit = (100, " pp") if metric == "balanced_accuracy" else (1, "")
            lines = [f"| Run − {args.reference} | " + " | ".join(label for _, label in CONDITIONS) + " |",
                     "|---|" + "---:|" * len(CONDITIONS)]
            for name in names:
                if name == args.reference:
                    continue
                cells = []
                for cond, _ in CONDITIONS:
                    diff = boot[(name, cond)][metric] - boot[(args.reference, cond)][metric]
                    point = (runs[name]["summary"]["test_metrics"][cond][metric]
                             - runs[args.reference]["summary"]["test_metrics"][cond][metric])
                    lo, hi = ci(diff)
                    star = "*" if lo > 0 or hi < 0 else ""
                    d = 1 if scale == 100 else 3     # pp with one decimal, AUROC with three
                    cells.append(f"{point * scale:+.{d}f}{star} [{lo * scale:+.{d}f}, {hi * scale:+.{d}f}]")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            sections += [f"## Paired difference vs `{args.reference}`: {title}{unit}", "\n".join(lines)]
    elif boot:
        print(f"Reference run {args.reference!r} not found; skipping the paired-difference tables")

    text = "\n\n".join(sections) + "\n"
    out = ROOT / "results" / f"comparison_holdout_{args.held_out}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
