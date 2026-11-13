"""Failure-case analysis for one held-out generator, using saved test scores only.

Reads results/<run>_holdout_<generator>/{summary.json,test_scores.csv} written by
train_resnet_generalization.py and dinov2_genimage_holdout.py; nothing is re-trained and
no threshold is re-fitted. Produces:

- results/analysis_holdout_<generator>.md
    1. File format / resolution of real vs generated images per GenImage generator
       (the JPEG-vs-PNG and fixed-size confound a detector can exploit).
    2. Real recall and generated recall per test condition: which class fails.
    3. Error overlap between two runs on the same images (are the failures shared?).
- reports/figures/<generator>_class_recall.png      recall of each class per condition
- reports/figures/<generator>_score_shift.png       score histograms, clean vs degraded
- reports/figures/<generator>_failures_<run>.png    most confident mistakes on clean images

    python scripts/failure_analysis.py --held-out imagenet_glide
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
TRAIN_ROOT = ROOT / "dataset" / "image" / "training" / "GenImage"
CONDITIONS = [
    ("clean", "Clean"),
    ("jpeg_q50", "JPEG Q50"),
    ("resize_128", "Resize 128"),
    ("resize_64", "Resize 64"),
    ("resize_32", "Resize 32"),
    ("resize_jpeg_128", "128 + JPEG 50"),
]
COLORS = ["#334155", "#f59e0b", "#2563eb", "#10b981", "#a855f7", "#ef4444"]


def load(run: str, held_out: str) -> dict:
    run_dir = RESULTS / f"{run}_holdout_{held_out}"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with open(run_dir / "test_scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_cond = {}
    for cond, _ in CONDITIONS:
        rs = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["path"])
        by_cond[cond] = {"path": np.array([r["path"] for r in rs]),
                         "real": np.array([r["label"] == "real" for r in rs]),
                         "score": np.array([float(r["p_real"]) for r in rs]),
                         "pred_real": np.array([r["prediction"] == "real" for r in rs])}
    return {"summary": summary, "scores": by_cond,
            "threshold": summary.get("threshold_selected_on_seen_validation")}


# ------------------------------------------------------------------ dataset confound

def format_table(per_folder: int) -> str:
    """Extension, JPEG quality proxy and shorter-side size of the first images per folder."""
    if not TRAIN_ROOT.is_dir():
        return "_GenImage training folder not found; skipped._"
    lines = ["| Generator | Class | Formats | Shorter side (median, range) | JPEG luminance DC quant. step (median) |",
             "|---|---|---|---|---:|"]
    for gen in sorted(p.name for p in TRAIN_ROOT.iterdir() if p.is_dir()):
        for cls, label in (("nature", "real"), ("ai", "generated")):
            files = sorted((TRAIN_ROOT / gen / "train" / cls).iterdir())[:per_folder]
            fmts, sides, steps = Counter(), [], []
            for f in files:
                with Image.open(f) as im:
                    fmts[im.format] += 1
                    sides.append(min(im.size))
                    q = getattr(im, "quantization", None)
                    if im.format == "JPEG" and q:
                        steps.append(q[0][0])
            fmt = ", ".join(f"{k} {v}" for k, v in fmts.most_common())
            step = f"{int(np.median(steps))}" if steps else "–"
            lines.append(f"| {gen} | {label} | {fmt} | {int(np.median(sides))} ({min(sides)}–{max(sides)}) | {step} |")
    return (f"First {per_folder} training images per folder. A luminance DC step of 1–2 means "
            "JPEG quality ≈ 95–100.\n\n" + "\n".join(lines))


# ------------------------------------------------------------------------ recalls

def recall_table(runs: dict) -> str:
    header = "| Run | " + " | ".join(f"{label} real / gen." for _, label in CONDITIONS) + " |"
    lines = [header, "|---|" + "---:|" * len(CONDITIONS)]
    for name, run in runs.items():
        cells = []
        for cond, _ in CONDITIONS:
            m = run["summary"]["test_metrics"][cond]
            cells.append(f"{m['real_correct'] / m['real_total']:.0%} / {m['fake_correct'] / m['fake_total']:.0%}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_recalls(runs: dict, held_out: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    width = 0.8 / len(runs)
    for ax, (key, title) in zip(axes, (("real", "Real images classified real"),
                                       ("fake", "Generated images classified generated"))):
        for i, (name, run) in enumerate(runs.items()):
            vals = [run["summary"]["test_metrics"][c][f"{key}_correct"] / run["summary"]["test_metrics"][c][f"{key}_total"]
                    for c, _ in CONDITIONS]
            ax.bar(x + (i - (len(runs) - 1) / 2) * width, vals, width, label=name, color=COLORS[i % len(COLORS)])
        ax.set_xticks(x, [label for _, label in CONDITIONS], rotation=20)
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Recall")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="outside lower center", ncol=len(runs))
    fig.suptitle(f"Which class fails? Held-out {held_out}, threshold fixed from seen-generator validation")
    out = FIGURES / f"{held_out}_class_recall.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


def plot_score_shift(runs: dict, held_out: str, conditions: list[str]) -> Path:
    fig, axes = plt.subplots(len(runs), len(conditions), figsize=(3.4 * len(conditions), 2.3 * len(runs)),
                             squeeze=False, constrained_layout=True)
    labels = dict(CONDITIONS)
    for r, (name, run) in enumerate(runs.items()):
        all_scores = np.concatenate([run["scores"][c]["score"] for c in conditions])
        lo, hi = np.percentile(all_scores, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        for c, cond in enumerate(conditions):
            ax = axes[r, c]
            d = run["scores"][cond]
            ax.hist(d["score"][d["real"]], bins=bins, alpha=0.6, color="#2563eb", label="real")
            ax.hist(d["score"][~d["real"]], bins=bins, alpha=0.6, color="#f59e0b", label="generated")
            if run["threshold"] is not None:
                ax.axvline(run["threshold"], color="black", lw=1, ls="--")
            ax.set_yticks([])
            if r == 0:
                ax.set_title(labels[cond])
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Score distributions (higher = more real); dashed line = validation-calibrated threshold")
    out = FIGURES / f"{held_out}_score_shift.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


def overlap_table(a: str, b: str, runs: dict) -> str:
    lines = [f"| Condition | Both correct | Only `{a}` wrong | Only `{b}` wrong | Both wrong |",
             "|---|---:|---:|---:|---:|"]
    for cond, label in CONDITIONS:
        da, db = runs[a]["scores"][cond], runs[b]["scores"][cond]
        assert np.array_equal(da["path"], db["path"]), "runs scored different images"
        wa, wb = da["pred_real"] != da["real"], db["pred_real"] != db["real"]
        lines.append(f"| {label} | {(~wa & ~wb).sum()} | {(wa & ~wb).sum()} | {(~wa & wb).sum()} | {(wa & wb).sum()} |")
    return "\n".join(lines)


def thumbnail(path: str, size: int = 160) -> Image.Image:
    with Image.open(ROOT / path) as im:
        return ImageOps.fit(im.convert("RGB"), (size, size))


def plot_failures(name: str, run: dict, held_out: str, n: int) -> Path:
    d = run["scores"]["clean"]
    thr = run["threshold"]
    real_wrong = np.flatnonzero(d["real"] & ~d["pred_real"])
    fake_wrong = np.flatnonzero(~d["real"] & d["pred_real"])
    real_wrong = real_wrong[np.argsort(d["score"][real_wrong])][:n]           # most confidently "generated"
    fake_wrong = fake_wrong[np.argsort(-d["score"][fake_wrong])][:n]          # most confidently "real"
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.6), squeeze=False, constrained_layout=True)
    for row, (idxs, title) in enumerate(((real_wrong, f"Real → called generated ({(d['real'] & ~d['pred_real']).sum()} total)"),
                                         (fake_wrong, f"Generated → called real ({(~d['real'] & d['pred_real']).sum()} total)"))):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(idxs):
                i = idxs[col]
                ax.imshow(thumbnail(d["path"][i]))
                ax.set_title(f"{d['score'][i]:.3f}", fontsize=8)
        axes[row, 0].text(-0.08, 0.5, title, transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=8)
    fig.suptitle(f"{name}: most confident clean-image mistakes on held-out {held_out} "
                 f"(score = real score, threshold {thr:.3f})", fontsize=10)
    out = FIGURES / f"{held_out}_failures_{name}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out", default="imagenet_glide")
    parser.add_argument("--runs", nargs="+",
                        default=["resnet", "resnet_jpeg", "resnet_robust", "dinov2_rigid",
                                 "dinov2_logreg_clean", "dinov2_mlp_clean_diff"],
                        help="run names (folder name without _holdout_<generator>)")
    parser.add_argument("--overlap", nargs=2, default=["resnet", "dinov2_mlp_clean_diff"],
                        help="two runs whose per-image errors are compared")
    parser.add_argument("--failure-runs", nargs="+", default=["resnet", "dinov2_mlp_clean_diff"])
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--format-sample", type=int, default=100, help="images per folder for the format table")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    wanted = list(dict.fromkeys([*args.runs, *args.overlap, *args.failure_runs]))
    runs = {}
    for name in wanted:
        if (RESULTS / f"{name}_holdout_{args.held_out}" / "summary.json").exists():
            runs[name] = load(name, args.held_out)
        else:
            print(f"skipping {name}: no results for {args.held_out}")
    shown = {n: runs[n] for n in args.runs if n in runs}

    figures = [plot_recalls(shown, args.held_out),
               plot_score_shift(shown, args.held_out, ["clean", "jpeg_q50", "resize_64", "resize_32"])]
    figures += [plot_failures(n, runs[n], args.held_out, args.n_examples) for n in args.failure_runs if n in runs]

    sections = [
        f"# Failure analysis: held-out `{args.held_out}`",
        "## 1. Format and resolution confound in GenImage", format_table(args.format_sample),
        "## 2. Recall of each class (real / generated) at the validation-calibrated threshold", recall_table(shown),
    ]
    a, b = args.overlap
    if a in runs and b in runs:
        sections += [f"## 3. Per-image error overlap: `{a}` vs `{b}`", overlap_table(a, b, runs)]
    sections += ["## Figures", "\n".join(f"- `{p.relative_to(ROOT).as_posix()}`" for p in figures)]
    out = RESULTS / f"analysis_holdout_{args.held_out}.md"
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
