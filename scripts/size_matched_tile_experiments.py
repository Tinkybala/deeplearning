"""Size-matched control for the native-resolution gain.

native_tile_experiments.py found that DINOv2 features from native-resolution 224 px tiles beat
the main runs' downscaled crop by about 4-5 pp. But in GenImage image size differs by class:
most real photos are about 375 x 500 (2 tiles), BigGAN/VQDM/ADM/GLIDE images 128-256 px (1
tile), SD v1.5/Wukong/Midjourney 512-1024 px (4 tiles). The native views keep that difference,
so the gain could be a size or content-scale cue rather than finer detail.

Here every image of both classes, in training and test, is first resized (bicubic) to shorter
side 448 (native_tile_features.py --size 448): all images then share one shorter side and
exactly 4 tiles, and a 224 px tile covers the same fraction of every image. If the gain is
finer detail, the 448 views should keep most of it; if it is a size cue, it should vanish.

Views (DINOv2 blocks 20 and 24, MLP probe of the main runs, clean seen-generator training,
leave-one-generator-out, clean seen-val threshold), tested on clean and Format q96:
    crop                 the main runs' downscaled crop (shorter side 256, centre 224)
    native centre / tile mean      native resolution (native_tile_experiments.py)
    448 crop / centre / tile mean  the same three views after resizing to shorter side 448
Also: real-photo recall by native size group (1, 2 or 4 native tiles), which shows whether a
view's decisions depend on a real photo's size.

    python scripts/size_matched_tile_experiments.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_comparison as bc  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402

GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS = ["clean", "fmt_q96"]
LABEL = {"clean": "Clean", "fmt_q96": "Format q96"}
LAYERS = [20, 24]
VIEWS = {   # name: (suffix, key prefix)
    "downscaled crop (main runs)": ("__tiles", "dino_crop"),
    "native centre": ("__tiles", "dino_center"),
    "native tile mean": ("__tiles", "dino_tilemean"),
    "448 crop": ("__tiles448", "dino_crop"),
    "448 centre": ("__tiles448", "dino_center"),
    "448 tile mean": ("__tiles448", "dino_tilemean"),
}
BASE = "downscaled crop (main runs)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    fstore = bc.FeatureStore()
    rng = np.random.default_rng(args.boot_seed)
    res, boots, groups = {}, {}, {}
    for g in GENERATORS:
        seen = [x for x in GENERATORS if x != g]
        fit_rec, val_rec = cpe.splits(args, seen, seen[0])
        val_new = bc.fa.shared_val(args, val_rec)
        test_rec = cpe.test_records(args, g)
        yf, yv, yt = (np.asarray([r[1] for r in rec]) for rec in (fit_rec, val_rec, test_rec))
        ntiles = fstore.gather(test_rec, "clean__tiles", ["n_tiles"]).ravel().astype(int)
        boot = {c: cpe.Boot(yt, args.bootstrap, rng) for c in CONDS}
        for l in LAYERS:
            for v, (suffix, prefix) in VIEWS.items():
                key = [f"{prefix}{l}"]
                fn = bc.fit_probe("MLP", fstore.gather(fit_rec, "clean" + suffix, key), yf, args)
                thr = cpe.best_threshold(yv, cpe.logit(fn(fstore.gather(val_new, "clean" + suffix, key))))[0]
                for c in CONDS:
                    s = cpe.logit(fn(fstore.gather(test_rec, c + suffix, key)))
                    res[(g, l, v, c)] = cpe.metrics(yt, s, thr)
                    boots[(g, l, v, c)] = boot[c](s, thr)
                    if l == 20:
                        for n in (1, 2, 4):
                            m = (yt == 1) & (ntiles == n)
                            groups.setdefault((v, c, n), []).append((s[m] >= thr))
        print(f"{SHORT[g]:10s} block-20 clean BA: " + "  ".join(
            f"{v} {res[(g, 20, v, 'clean')]['balanced_accuracy']:.3f}" for v in VIEWS), flush=True)
    report(args, res, boots, groups)


def report(args, res, boots, groups) -> None:
    gens, views = GENERATORS, list(VIEWS)

    def mean(l, v, c, metric="balanced_accuracy"):
        return float(np.mean([res[(g, l, v, c)][metric] for g in gens]))

    lines = ["# Size-matched control for the native-resolution gain",
             "DINOv2 MLP probes (main runs' recipe) fit on clean seen-generator training images, one view each; "
             "leave-one-generator-out, clean seen-val threshold. **native** views keep each image's own size, which "
             "differs by class in GenImage. **448** views first resize every image, of both classes, in training and "
             "test, to shorter side 448 (bicubic): one shorter side and exactly 4 tiles for every image. **crop** = "
             "shorter side 256, centre 224; **centre** = central 224 px; **tile mean** = mean CLS over the 224 px tiles. "
             "**Format q96** = both classes re-saved as JPEG q96 at native size (before any resizing). Mean over the 7 holdouts."]
    for metric, title, f in (("balanced_accuracy", "Balanced accuracy", lambda x: f"{x:.1%}"),
                             ("auroc_real_positive", "AUROC", lambda x: f"{x:.3f}")):
        rows = ["| Block | View | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|---|" + "---:|" * len(CONDS)]
        for l in LAYERS:
            for v in views:
                rows.append(f"| {l} | {v} | " + " | ".join(f(mean(l, v, c, metric)) for c in CONDS) + " |")
        lines += [f"## {title}", "\n".join(rows)]
    rows = ["| Block | View − downscaled crop | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|---|" + "---:|" * len(CONDS)]
    for l in LAYERS:
        for v in views[1:]:
            cells = []
            for c in CONDS:
                point = np.mean([res[(g, l, v, c)]["balanced_accuracy"] - res[(g, l, BASE, c)]["balanced_accuracy"] for g in gens])
                rep = np.mean([boots[(g, l, v, c)]["balanced_accuracy"] - boots[(g, l, BASE, c)]["balanced_accuracy"]
                               for g in gens], axis=0)
                cells.append(cpe.fmt_diff(point, rep, 100, 1))
            rows.append(f"| {l} | {v} | " + " | ".join(cells) + " |")
    lines += [f"## Difference from the downscaled crop, balanced accuracy (pp), paired bootstrap ({args.bootstrap} "
              "replicates per holdout); * = 95% interval excludes 0", "\n".join(rows)]
    for c in CONDS:
        rows = ["| Held-out generator | " + " | ".join(views) + " |", "|---|" + "---:|" * len(views)]
        for g in gens:
            vals = [res[(g, 20, v, c)]["balanced_accuracy"] for v in views]
            rows.append(f"| {SHORT[g]} | " + " | ".join(f"**{x:.1%}**" if x == max(vals) else f"{x:.1%}" for x in vals) + " |")
        lines += [f"## Block 20, {LABEL[c]}, per held-out generator", "\n".join(rows)]
    rows = ["| View | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
    for v in views:
        rows.append(f"| {v} | " + " | ".join(
            f"{mean(20, v, c, 'real_recall'):.0%} / {mean(20, v, c, 'generated_recall'):.0%}" for c in CONDS) + " |")
    lines += ["## Class recall, block 20 (real / generated)", "\n".join(rows)]
    rows = ["| View | " + " | ".join(f"{LABEL[c]}, {n} tile{'s' if n > 1 else ''} (n = {sum(len(x) for x in groups[(views[0], c, n)])})"
                                     for c in CONDS for n in (1, 2, 4)) + " |", "|---|" + "---:|" * (3 * len(CONDS))]
    for v in views:
        rows.append(f"| {v} | " + " | ".join(f"{np.concatenate(groups[(v, c, n)]).mean():.1%}"
                                             for c in CONDS for n in (1, 2, 4)) + " |")
    lines += ["## Real-photo recall by native size, block 20 (pooled over the 7 holdouts). 1 tile = both sides under "
              "448 px, 2 tiles = one side at least 448 px, 4 tiles = both", "\n".join(rows)]
    data = {f"{g}|{l}|{v}|{c}": x for (g, l, v, c), x in res.items()}
    data |= {f"real_recall_by_size|{v}|{c}|{n}": float(np.concatenate(x).mean()) for (v, c, n), x in groups.items()}
    cpe.write_report("size_matched_tile_experiments", lines, data)


if __name__ == "__main__":
    main()
