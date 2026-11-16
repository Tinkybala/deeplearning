"""Figures for the final report (report/figures/*.pdf), from the result files in results/.

    python report/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
RES = ROOT / "results"

# Reference categorical palette, fixed order; marker shapes as a second encoding.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

GENS = ["imagenet_ai_0419_biggan", "imagenet_ai_0419_vqdm", "imagenet_ai_0424_sdv5", "imagenet_ai_0424_wukong",
        "imagenet_ai_0508_adm", "imagenet_glide", "imagenet_midjourney"]
SHORT = ["BigGAN", "VQDM", "SD v1.5", "Wukong", "ADM", "GLIDE", "Midjourney"]
STD = ["clean", "jpeg_q50", "resize_128", "resize_64", "resize_32", "resize_jpeg_128"]
CONDS = STD + ["fmt_r256_q96"]
CLABEL = ["Clean", "JPEG Q50", "Resize 128", "Resize 64", "Resize 32", "128 + JPEG 50", "Format 256+q96"]
DETECTORS = [  # label, results file, method key
    ("Fine-tuned ResNet-18", "resnet_dinov2_comparison", "ResNet-18 squash224"),
    ("DINOv2 final layer, MLP (main runs)", "resnet_dinov2_comparison", "DINOv2 MLP"),
    ("Final detector: DINOv2 block 20, aug.", "dinov3_layer_experiments", "DINOv2 L20 MLP [clean+both]"),
    ("DINOv2 + DINOv3 fusion", "dinov3_layer_experiments", "fusion DINOv2 L20 + DINOv3 L24 [clean+both]"),
]

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.8, "pdf.fonttype": 42})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def ba(data, g, m, c):
    return 100 * data[f"{g}|{m}|{c}"]["balanced_accuracy"]


def style(ax, xlabel):
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.axvline(50, color=AXIS, linewidth=0.8, linestyle=(0, (3, 2)))


def dot_plot(rows, values, path, xlabel, xlim):
    """Rows top to bottom; one marker per detector, slightly offset vertically."""
    fig, ax = plt.subplots(figsize=(6.3, 0.42 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    for i, (label, *_ ) in enumerate(DETECTORS):
        off = (i - 1.5) * 0.15
        ax.scatter(values[i], y + off, s=34, marker=MARKERS[i], color=SERIES[i], edgecolor="white",
                   linewidth=0.6, zorder=3, label=label)
    for yy in y:
        ax.axhline(yy, color=GRID, linewidth=0.4, zorder=1)
    ax.set_yticks(y, rows)
    ax.set_xlim(*xlim)
    style(ax, xlabel)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8,
              handletextpad=0.3, columnspacing=1.2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_conditions():
    vals = []
    for _, f, m in DETECTORS:
        d = load(f)
        vals.append([np.mean([ba(d, g, m, c) for g in GENS]) for c in CONDS])
    dot_plot(CLABEL, vals, OUT / "conditions.pdf", "Balanced accuracy (%), mean over the 7 held-out generators", (45, 95))


def fig_generators():
    vals = []
    for _, f, m in DETECTORS:
        d = load(f)
        vals.append([np.mean([ba(d, g, m, c) for c in STD]) for g in GENS])
    dot_plot(SHORT, vals, OUT / "generators.pdf",
             "Balanced accuracy (%) on the held-out generator, mean over the six standard conditions", (45, 95))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fig_cifake():
    """Score histograms on CIFAKE, real vs generated, with each detector's shipped threshold."""
    ext = RES / "external"
    panels = []
    for name, col, model in (("cifake", "p_real", "final_detector_dinov2_l20_mlp.pt"),
                             ("cifake_fusion", "fused_score", "fusion_detector_dinov2_l20_dinov3_l24.pt")):
        rows = list(csv.DictReader(open(ext / f"{name}_scores.csv")))
        s = logit([float(r[col]) for r in rows])
        y = np.array([r["label"] == "real" for r in rows])
        thr = json.loads((ext / f"{name}.json").read_text())["threshold"] if "threshold" in json.loads(
            (ext / f"{name}.json").read_text()) else json.loads((ext / f"{name}.json").read_text())["threshold_p_real"]
        panels.append((s, y, float(logit([thr])[0])))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    titles = ["Final detector (DINOv2 block 20)", "DINOv2 + DINOv3 fusion"]
    xlabels = ["logit P(real)", "fused score (standardised logit)"]
    for ax, (s, y, t), title, xl in zip(axes, panels, titles, xlabels):
        lo, hi = np.percentile(s, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax.hist(s[y], bins=bins, color=SERIES[0], alpha=0.75, label="real (CIFAR-10)", edgecolor="white", linewidth=0.3)
        ax.hist(s[~y], bins=bins, color=SERIES[1], alpha=0.75, label="generated (SD v1.4)", edgecolor="white", linewidth=0.3)
        ax.axvline(t, color=INK, linewidth=1.0)
        ax.text(t, ax.get_ylim()[1] * 0.97, " threshold", color=INK2, fontsize=7.5, va="top")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel(xl)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "cifake_scores.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_conditions()
    fig_generators()
    fig_cifake()
    print("Saved", *sorted(p.name for p in OUT.glob("*.pdf")))
