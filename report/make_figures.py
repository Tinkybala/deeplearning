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


def _project():
    """Project modules (scripts/), imported only by the figures that need them."""
    import sys
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    import backbone_comparison as bc
    import cached_probe_experiments as cpe
    args = type("A", (), {})()
    cpe.set_defaults(args)
    return bc, cpe, args


TEST = ROOT / "dataset" / "image" / "test" / "GenImage"
EXAMPLE_INDEX = 3   # which file (sorted by name) of each test folder to show


def fig_examples():
    """One real photo and one image per generator, as the detector sees them, under three conditions."""
    from torchvision import transforms
    bc, cpe, args = _project()
    resnet = bc.dh.resnet
    view = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    cols = [("Real photo", sorted((TEST / GENS[5] / "test" / "nature").iterdir())[EXAMPLE_INDEX])]
    cols += [(s, sorted((TEST / g / "test" / "ai").iterdir())[EXAMPLE_INDEX]) for g, s in zip(GENS, SHORT)]
    conds = [("clean", "Clean"), ("jpeg_q50", "JPEG Q50"), ("resize_32", "Resize 32")]
    fig, axes = plt.subplots(len(conds), len(cols), figsize=(6.3, 2.75))
    for j, (name, path) in enumerate(cols):
        for i, (c, label) in enumerate(conds):
            img = resnet.ImageRecords([(path, 0, "")], view, c)[0][0]
            ax = axes[i, j]
            ax.imshow(np.asarray(img), interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(name, fontsize=8, color=INK, pad=3)
            if j == 0:
                ax.set_ylabel(label, fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.92, bottom=0.01, wspace=0.04, hspace=0.06)
    fig.savefig(OUT / "examples.pdf", dpi=300)
    plt.close(fig)


def fig_layers():
    """Balanced accuracy by DINOv2 block: seen-generator validation vs unseen generators, by training."""
    d = load("augmented_layer_selection")
    blocks = [8, 12, 16, 20, 24]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    lines = [("Seen generators (validation), clean", None),
             ("Unseen generator, clean", "clean"),
             ("Unseen generator, JPEG Q50", "jpeg_q50"),
             ("Unseen generator, resize 32", "resize_32")]
    for ax, train, title in zip(axes, ("clean", "clean+both"), ("Trained on clean images", "Trained on clean + augmented images")):
        for i, (label, c) in enumerate(lines):
            ys = []
            for b in blocks:
                m = f"DINOv2 L{b} MLP [{train}]"
                if c is None:
                    ys.append(100 * np.mean([d[f"seenval|{g}"][m] for g in GENS]))
                else:
                    ys.append(100 * np.mean([d[f"{g}|{m}|{c}"]["balanced_accuracy"] for g in GENS]))
            ax.plot(blocks, ys, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6, label=label,
                    linestyle=(0, (4, 2)) if c is None else "solid")
        ax.axvline(20, color=AXIS, linewidth=0.8, zorder=0)
        ax.text(20, 47, " block 20", color=MUTED, fontsize=7.5, va="bottom")
        ax.set_xticks(blocks)
        ax.set_xlabel("DINOv2 block")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylim(45, 100)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Balanced accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "layers.pdf")
    plt.close(fig)


def fig_drift():
    """Block-20 MLP scores relative to the fixed threshold, clean vs JPEG Q50, pooled over the 7 folds."""
    bc, cpe, args = _project()
    fstore = bc.FeatureStore()
    key = ["dino_cls20"]
    out = {}
    for g in GENS:
        seen = [x for x in GENS if x != g]
        fit_rec, val_rec = cpe.splits(args, seen, seen[0])
        val_new = bc.fa.shared_val(args, val_rec)
        test_rec = cpe.test_records(args, g)
        yf, yv, yt = (np.asarray([r[1] for r in rec]) for rec in (fit_rec, val_rec, test_rec))
        for train in ("clean", "clean+both"):
            X, y = fstore.gather(fit_rec, "clean", key), yf
            if train == "clean+both":
                X, y = np.vstack([X, fstore.gather(fit_rec, "clean__aug-both", key)]), np.concatenate([yf, yf])
            fn = bc.fit_probe("MLP", X, y, args)
            thr = cpe.best_threshold(yv, cpe.logit(fn(fstore.gather(val_new, "clean", key))))[0]
            for c in ("clean", "jpeg_q50"):
                s = cpe.logit(fn(fstore.gather(test_rec, c, key))) - thr
                out.setdefault((train, c), []).append((s, yt))
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.9), sharex=True)
    rows = [("clean", "Trained on clean images"), ("clean+both", "Trained on clean + augmented images")]
    cols = [("clean", "Clean test images"), ("jpeg_q50", "JPEG Q50 test images")]
    bins = np.linspace(-16, 16, 65)
    for i, (train, rlabel) in enumerate(rows):
        for j, (c, clabel) in enumerate(cols):
            s = np.concatenate([x[0] for x in out[(train, c)]])
            y = np.concatenate([x[1] for x in out[(train, c)]]) == 1
            ba = 100 * np.mean([0.5 * ((x[0][x[1] == 1] >= 0).mean() + (x[0][x[1] == 0] < 0).mean()) for x in out[(train, c)]])
            ax = axes[i, j]
            ax.hist(np.clip(s[y], -16, 16), bins=bins, color=SERIES[0], alpha=0.75, label="real", edgecolor="white", linewidth=0.3)
            ax.hist(np.clip(s[~y], -16, 16), bins=bins, color=SERIES[1], alpha=0.75, label="generated", edgecolor="white", linewidth=0.3)
            ax.axvline(0, color=INK, linewidth=1.0)
            ax.text(0.02, 0.95, f"BA {ba:.1f}%", transform=ax.transAxes, fontsize=8, color=INK, va="top")
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(clabel, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(rlabel, fontsize=8, color=INK2)
            if i == 1:
                ax.set_xlabel("logit P(real) − threshold")
            print(f"  drift {train:10s} {c:9s} BA {ba:.1f}%")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "drift.pdf")
    plt.close(fig)


def fig_sizes():
    """Shorter side of the test images by class: the size and format confound."""
    from PIL import Image
    rows = []
    real = sorted((TEST / GENS[0] / "test" / "nature").iterdir())
    sides = []
    for p in real:
        with Image.open(p) as im:
            sides.append(min(im.size))
    rows.append(("Real photos (JPEG)", np.asarray(sides)))
    for g, s in zip(GENS, SHORT):
        sides = []
        for p in sorted((TEST / g / "test" / "ai").iterdir()):
            with Image.open(p) as im:
                sides.append(min(im.size))
        rows.append((f"{s} (PNG)", np.asarray(sides)))
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    rng = np.random.default_rng(0)
    y = np.arange(len(rows))[::-1]
    for k, ((label, v), yy) in enumerate(zip(rows, y)):
        color = SERIES[0] if k == 0 else SERIES[1]
        ax.scatter(v, yy + rng.uniform(-0.25, 0.25, len(v)), s=4, color=color, alpha=0.35, linewidth=0)
        lo, hi = v.min(), v.max()
        note = f"{lo}" if lo == hi else f"{np.median(v):.0f} median ({lo}–{hi})"
        ax.text(1.01, yy, note, fontsize=7.5, color=INK2, va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xticks([128, 256, 512, 1024, 2048], ["128", "256", "512", "1024", "2048"])
    ax.set_xlim(80, 2100)
    ax.set_xlabel("Shorter side of the test image (pixels, log scale)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(OUT / "sizes.pdf", dpi=300)
    plt.close(fig)
    print("  sizes:", {r[0]: (int(r[1].min()), int(np.median(r[1])), int(r[1].max())) for r in rows})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_conditions()
    fig_generators()
    fig_cifake()
    fig_examples()
    fig_layers()
    fig_sizes()
    fig_drift()
    print("Saved", *sorted(p.name for p in OUT.glob("*.pdf")))
