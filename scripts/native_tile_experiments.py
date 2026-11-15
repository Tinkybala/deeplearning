"""Does native resolution help? Downscaled crop vs native centre crop vs native tile mean.

Uses native_tile_features.py, whose three views of every image come from one pass and one
pipeline (its crop view reproduces backbone_features.py exactly). For DINOv2 blocks 20 and 24
it fits the main runs' MLP probe on each view of the clean seen-generator training images
(leave-one-generator-out, clean seen-val threshold) and tests on the held-out generator under
clean, JPEG Q50 and Format q96 (both classes re-saved as JPEG q96 at native size, which
removes the PNG-vs-JPEG difference). Also: the crop and tile-mean MLPs' logits averaged
(multi-scale), standardised on clean seen-val. Training is clean only; the comparison is
between views, not against the augmented final detector.

    python scripts/native_tile_experiments.py
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
import native_tile_features as ntf  # noqa: E402

GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS = ntf.TEST_CONDS
LABEL = bc.LABEL | {"fmt_q96": "Format q96"}
VIEWS = {"crop": "downscaled crop (main runs)", "center": "native centre crop", "tilemean": "native tile mean"}
LAYERS = ntf.LAYERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    fstore = bc.FeatureStore()
    rng = np.random.default_rng(args.boot_seed)
    res, boots = {}, {}
    S = ntf.SUFFIX
    for g in GENERATORS:
        seen = [x for x in GENERATORS if x != g]
        fit_rec, val_rec = cpe.splits(args, seen, seen[0])
        val_new = bc.fa.shared_val(args, val_rec)
        test_rec = cpe.test_records(args, g)
        yf, yv, yt = (np.asarray([r[1] for r in rec]) for rec in (fit_rec, val_rec, test_rec))
        boot = {c: cpe.Boot(yt, args.bootstrap, rng) for c in CONDS}
        for l in LAYERS:
            sc = {}
            for v in VIEWS:
                key = [f"dino_{v}{l}"]
                fn = bc.fit_probe("MLP", fstore.gather(fit_rec, "clean" + S, key), yf, args)
                sc[v] = {"val": cpe.logit(fn(fstore.gather(val_new, "clean" + S, key)))}
                sc[v] |= {c: cpe.logit(fn(fstore.gather(test_rec, c + S, key))) for c in CONDS}
            z = lambda v, k: (sc[v][k] - sc[v]["val"].mean()) / sc[v]["val"].std()  # noqa: E731
            sc["crop + tilemean"] = {k: 0.5 * (z("crop", k) + z("tilemean", k)) for k in ["val"] + CONDS}
            for v, s in sc.items():
                thr = cpe.best_threshold(yv, s["val"])[0]
                for c in CONDS:
                    res[(g, l, v, c)] = cpe.metrics(yt, s[c], thr)
                    boots[(g, l, v, c)] = boot[c](s[c], thr)
        print(f"{SHORT[g]:10s} block-20 clean BA: " + "  ".join(
            f"{v} {res[(g, 20, v, 'clean')]['balanced_accuracy']:.3f}" for v in list(VIEWS) + ["crop + tilemean"]), flush=True)
    report(args, res, boots)


def report(args, res, boots) -> None:
    gens = GENERATORS
    views = list(VIEWS) + ["crop + tilemean"]

    def mean(l, v, c, metric="balanced_accuracy"):
        return float(np.mean([res[(g, l, v, c)][metric] for g in gens]))

    lines = ["# Native resolution: downscaled crop vs native centre crop vs native tile mean",
             "DINOv2 MLP probes (main runs' recipe) fit on clean seen-generator training images, one view each; "
             "leave-one-generator-out, clean seen-val threshold. **crop** = shorter side 256, centre 224 (main runs); "
             "**center** = central 224 px at native resolution; **tilemean** = mean CLS over up to four native-resolution "
             "224 px tiles; **crop + tilemean** = the two MLPs' standardised logits averaged. **Format q96** = both classes "
             "re-saved as JPEG q96 at native size. Mean over the 7 holdouts."]
    for metric, title, f in (("balanced_accuracy", "Balanced accuracy", lambda x: f"{x:.1%}"),
                             ("auroc_real_positive", "AUROC", lambda x: f"{x:.3f}")):
        rows = ["| Block | View | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|---|" + "---:|" * len(CONDS)]
        for l in LAYERS:
            for v in views:
                rows.append(f"| {l} | {VIEWS.get(v, v)} | " + " | ".join(f(mean(l, v, c, metric)) for c in CONDS) + " |")
        lines += [f"## {title}", "\n".join(rows)]
    rows = ["| Block | View − crop | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|---|" + "---:|" * len(CONDS)]
    for l in LAYERS:
        for v in views[1:]:
            cells = []
            for c in CONDS:
                point = np.mean([res[(g, l, v, c)]["balanced_accuracy"] - res[(g, l, "crop", c)]["balanced_accuracy"] for g in gens])
                rep = np.mean([boots[(g, l, v, c)]["balanced_accuracy"] - boots[(g, l, "crop", c)]["balanced_accuracy"]
                               for g in gens], axis=0)
                cells.append(cpe.fmt_diff(point, rep, 100, 1))
            rows.append(f"| {l} | {VIEWS.get(v, v)} | " + " | ".join(cells) + " |")
    lines += [f"## Difference from the downscaled crop, balanced accuracy (pp), paired bootstrap ({args.bootstrap} "
              "replicates per holdout); * = 95% interval excludes 0", "\n".join(rows)]
    for c in CONDS:
        rows = ["| Held-out generator | " + " | ".join(VIEWS.get(v, v) for v in views) + " |", "|---|" + "---:|" * len(views)]
        for g in gens:
            vals = [res[(g, 20, v, c)]["balanced_accuracy"] for v in views]
            rows.append(f"| {SHORT[g]} | " + " | ".join(f"**{x:.1%}**" if x == max(vals) else f"{x:.1%}" for x in vals) + " |")
        lines += [f"## Block 20, {LABEL[c]}, per held-out generator", "\n".join(rows)]
    rows = ["| Block | View | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|---|" + "---:|" * len(CONDS)]
    for v in views:
        rows.append(f"| 20 | {VIEWS.get(v, v)} | " + " | ".join(
            f"{mean(20, v, c, 'real_recall'):.0%} / {mean(20, v, c, 'generated_recall'):.0%}" for c in CONDS) + " |")
    lines += ["## Class recall, block 20 (real / generated)", "\n".join(rows)]
    cpe.write_report("native_tile_experiments", lines, {f"{g}|{l}|{v}|{c}": x for (g, l, v, c), x in res.items()})


if __name__ == "__main__":
    main()
