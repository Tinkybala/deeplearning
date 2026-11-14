"""Is the frozen-DINOv2 detector reading the file format instead of the generator? (E1 / E2)

In tiny-GenImage every real photo is a JPEG (almost all quality 96, 4:4:4 chroma, shorter
side ~375 px) while every generated image is a lossless PNG at a fixed size (128 px BigGAN,
256 px VQDM/ADM/GLIDE, 512 px SD/Wukong, 1024 px Midjourney). A detector can therefore
separate the classes by "was this JPEG-compressed?" without learning anything about
generators. Two new test conditions put *both* classes through the same format pipeline:

    fmt_q96        decode -> JPEG quality 96, 4:4:4 (the real photos' own setting) at the
                   native size -> the usual eval preprocessing
    fmt_r256_q96   decode -> resize shorter side to 256 (bicubic) -> JPEG q96 4:4:4 -> eval
                   preprocessing (whose own resize to 256 is then a no-op). The JPEG grid now
                   sits at the same scale for both classes, so neither compression nor size
                   tells them apart.

Real photos are compressed a second time (double JPEG); generated images once. That residual
is small at q96 but is stated wherever the results are reported.

E1 (b1)  format-matched test set. The probes of the main leave-one-generator-out runs (fit
         on clean embeddings, threshold from the clean seen-generator validation split) are
         evaluated on the held-out generator's test images under the two format conditions.
         A drop from clean means part of the clean accuracy was a format shortcut.
E2 (b2)  format-aligned training. The same probes are refit on seen-generator images that
         went through fmt_r256_q96 (both classes), threshold from the fmt_r256_q96
         seen-generator validation split, and evaluated on all eight test conditions.

The seen-generator validation fakes are the files in training/<g>/val/ai, which are
byte-identical to test/<g>/test/ai; the latter folders are used so their fmt embeddings are
shared with the E1 test sets. The held-out generator's own folders are never in fit or val.

    python scripts/format_alignment_experiments.py b1
    python scripts/format_alignment_experiments.py b2     # needs b1's embeddings (or makes them)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cached_probe_experiments as cpe  # noqa: E402
import dinov2_genimage_holdout as dh  # noqa: E402

resnet = dh.resnet
GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
FORMAT = {"fmt_q96": (None, 96), "fmt_r256_q96": (256, 96)}
TRAIN_FORMAT = "fmt_r256_q96"
TEST_CONDITIONS = dh.CONDITIONS + list(FORMAT)
LABEL = cpe.COND_LABEL | {"fmt_q96": "Format q96", "fmt_r256_q96": "Format 256+q96"}


class FormatRecords(resnet.ImageRecords):
    """resnet.ImageRecords plus the two format-alignment conditions."""

    def __getitem__(self, idx):
        if self.condition not in FORMAT:
            return super().__getitem__(idx)
        side, quality = FORMAT[self.condition]
        path, label, generator = self.records[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if side:
                image = resnet.resize_shorter_side(image, side)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, subsampling=0)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
            return self.transform(image), label, generator, str(path)


resnet.ImageRecords = FormatRecords      # EmbeddingStore._embed builds resnet.ImageRecords


def shared_val(args, val_rec: list) -> list:
    """Point seen-generator val fakes at the identical files under the test root."""
    out = []
    for path, label, gen in val_rec:
        p = Path(path)
        if label == 0:
            p = args.test_data_root / gen / args.test_split / "ai" / p.name
            assert p.exists(), p
        out.append((p, label, gen))
    return out


def evaluate(args, store, train: str) -> tuple[dict, dict]:
    """Fit every detector per holdout on `train` ('clean' or TRAIN_FORMAT), test on all conditions."""
    results, boots = {}, {}
    rng = np.random.default_rng(args.boot_seed)
    for held_out in GENERATORS:
        seen = [g for g in GENERATORS if g != held_out]
        fit_rec, val_rec = cpe.splits(args, seen, seen[0])       # same split as the main LOGO runs
        if train != "clean":
            val_rec = shared_val(args, val_rec)
        t0 = time.time()
        fit, val = store.gather(fit_rec, train, "none"), store.gather(val_rec, train, "none")
        tests = {c: store.gather(cpe.test_records(args, held_out), c, "none") for c in TEST_CONDITIONS}
        boot = {c: cpe.Boot(tests[c]["label"].astype(int), args.bootstrap, rng) for c in TEST_CONDITIONS}
        for det in dh.DETECTORS:
            score_fn, _ = dh.fit_detector(det, fit, args, held_out)
            thr, _ = cpe.best_threshold(val["label"].astype(int), score_fn(val))
            for c in TEST_CONDITIONS:
                s = score_fn(tests[c])
                results[(held_out, det, c)] = cpe.metrics(tests[c]["label"].astype(int), s, thr)
                boots[(held_out, det, c)] = boot[c](s, thr)
        if train == "clean":      # reproduction check against the saved main runs
            for det in dh.DETECTORS:
                run = f"dinov2_{det.replace('+', '_')}_holdout_{held_out}"
                saved = json.loads((ROOT / "results" / run / "summary.json").read_text())["test_metrics"]["clean"]
                got = results[(held_out, det, "clean")]["balanced_accuracy"]
                if abs(saved["balanced_accuracy"] - got) > 1e-9:
                    print(f"WARNING {run}: clean BA {got:.4f} differs from the saved run's "
                          f"{saved['balanced_accuracy']:.4f}", flush=True)
        mlp = {c: results[(held_out, "mlp_clean+diff", c)]["balanced_accuracy"] for c in ("clean", *FORMAT)}
        print(f"{train} -> {SHORT[held_out]:10s} mlp BA " + "  ".join(f"{c} {v:.3f}" for c, v in mlp.items())
              + f"  ({time.time() - t0:.0f}s, {store.embedded} images embedded so far)", flush=True)
    return results, boots


def mean_table(results, conds, metric, f, detectors=dh.DETECTORS, tag=lambda d: d) -> list[str]:
    rows = ["| Detector | " + " | ".join(LABEL[c] for c in conds) + " |", "|---|" + "---:|" * len(conds)]
    for det in detectors:
        cells = []
        for c in conds:
            vals = [results[(g, det, c)][metric] for g in GENERATORS]
            cells.append(f.format(np.mean(vals)) + f" [{f.format(min(vals))}]".replace("%]", "]"))
        rows.append(f"| {tag(det)} | " + " | ".join(cells) + " |")
    return rows


def diff_cells(res_a, boot_a, res_b, boot_b, det, conds, metric, scale, d) -> list[str]:
    """Mean over holdouts of (b - a), paired bootstrap per holdout."""
    cells = []
    for ca, cb in conds:
        point = np.mean([res_b[(g, det, cb)][metric] - res_a[(g, det, ca)][metric] for g in GENERATORS])
        reps = np.mean([boot_b[(g, det, cb)][metric] - boot_a[(g, det, ca)][metric] for g in GENERATORS], axis=0)
        cells.append(cpe.fmt_diff(point, reps, scale, d))
    return cells


def recall_table(results, conds) -> list[str]:
    rows = ["| Detector | " + " | ".join(f"{LABEL[c]} real / gen." for c in conds) + " |",
            "|---|" + "---:|" * len(conds)]
    for det in dh.DETECTORS:
        cells = [f"{np.mean([results[(g, det, c)]['real_recall'] for g in GENERATORS]):.0%} / "
                 f"{np.mean([results[(g, det, c)]['generated_recall'] for g in GENERATORS]):.0%}" for c in conds]
        rows.append(f"| {det} | " + " | ".join(cells) + " |")
    return rows


def per_generator(results, det, conds, metric, f) -> list[str]:
    rows = ["| Held-out generator | " + " | ".join(LABEL[c] for c in conds) + " |", "|---|" + "---:|" * len(conds)]
    rows += [f"| {SHORT[g]} | " + " | ".join(f.format(results[(g, det, c)][metric]) for c in conds) + " |"
             for g in GENERATORS]
    return rows


def run_b1(args, store) -> tuple[dict, dict]:
    res, boot = evaluate(args, store, "clean")
    conds = ["clean", "fmt_q96", "fmt_r256_q96", "jpeg_q50"]
    lines = ["# E1: format-matched test set",
             "Probes from the main leave-one-generator-out runs (fit on clean seen-generator embeddings, "
             "threshold from the clean seen-generator validation split), tested on the held-out generator with "
             "**both** classes put through the same format pipeline. **Format q96** = re-saved as JPEG quality 96 "
             "4:4:4 (the real photos' own setting) at native size; **Format 256+q96** = resized to shorter side 256, "
             "then JPEG q96, so compression and size no longer differ between classes. Real photos are thereby "
             "JPEG-compressed twice, generated images once. JPEG Q50 (both classes, from the main runs) is shown "
             "for reference. Mean over the 7 held-out generators, worst generator in brackets."]
    for metric, title, f in [("balanced_accuracy", "Balanced accuracy", "{:.1%}"), ("auroc_real_positive", "AUROC", "{:.3f}")]:
        lines += [f"## {title}", "\n".join(mean_table(res, conds, metric, f))]
    for metric, title, scale, d in [("balanced_accuracy", "balanced accuracy (pp)", 100, 1), ("auroc_real_positive", "AUROC", 1, 3)]:
        rows = ["| Detector | Format q96 − clean | Format 256+q96 − clean |", "|---|---:|---:|"]
        rows += [f"| {det} | " + " | ".join(diff_cells(res, boot, res, boot, det,
                                                       [("clean", "fmt_q96"), ("clean", "fmt_r256_q96")],
                                                       metric, scale, d)) + " |" for det in dh.DETECTORS]
        lines += [f"## Change in {title}, 7-generator mean, paired bootstrap ({args.bootstrap} replicates "
                  "per holdout); * = 95% interval excludes 0", "\n".join(rows)]
    lines += ["## Class recall (7-generator mean): real / generated", "\n".join(recall_table(res, conds[:3]))]
    lines += ["## mlp_clean+diff AUROC per held-out generator",
              "\n".join(per_generator(res, "mlp_clean+diff", conds[:3], "auroc_real_positive", "{:.3f}"))]
    cpe.write_report("e1_format_matched_test", lines, {f"{g}|{d}|{c}": v for (g, d, c), v in res.items()})
    return res, boot


def run_b2(args, store) -> None:
    res_c, boot_c = run_b1(args, store)
    res_f, boot_f = evaluate(args, store, TRAIN_FORMAT)
    lines = ["# E2: format-aligned training",
             "**clean-trained** = the main runs' probes. **aligned-trained** = the same probes refit on seen-generator "
             "images whose real *and* generated images went through Format 256+q96 (resize shorter side to 256, "
             "JPEG q96 4:4:4), threshold from the seen-generator validation split under the same pipeline. Both are "
             "tested on all eight conditions of the held-out generator (the six standard ones are the unaligned "
             "files, exactly as in the main runs). RIGID has no trained part; only its threshold changes. Mean over "
             "the 7 held-out generators, worst generator in brackets."]
    for metric, title, f in [("balanced_accuracy", "Balanced accuracy", "{:.1%}"), ("auroc_real_positive", "AUROC", "{:.3f}")]:
        rows = mean_table(res_c, TEST_CONDITIONS, metric, f, tag=lambda d: f"{d} (clean-trained)")
        rows += mean_table(res_f, TEST_CONDITIONS, metric, f, tag=lambda d: f"{d} (aligned-trained)")[2:]
        lines += [f"## {title}", "\n".join(rows)]
    for metric, title, scale, d in [("balanced_accuracy", "balanced accuracy (pp)", 100, 1), ("auroc_real_positive", "AUROC", 1, 3)]:
        rows = ["| Detector | " + " | ".join(LABEL[c] for c in TEST_CONDITIONS) + " |",
                "|---|" + "---:|" * len(TEST_CONDITIONS)]
        rows += [f"| {det} | " + " | ".join(diff_cells(res_c, boot_c, res_f, boot_f, det,
                                                       [(c, c) for c in TEST_CONDITIONS], metric, scale, d)) + " |"
                 for det in dh.DETECTORS]
        lines += [f"## Aligned-trained − clean-trained, {title}, 7-generator mean, paired bootstrap "
                  f"({args.bootstrap} replicates per holdout); * = 95% interval excludes 0", "\n".join(rows)]
    lines += ["## Class recall of aligned-trained probes (7-generator mean): real / generated",
              "\n".join(recall_table(res_f, ["clean", "jpeg_q50", "fmt_r256_q96"]))]
    rows = ["| Held-out generator | Clean (clean-tr.) | Clean (aligned-tr.) | 256+q96 (clean-tr.) | 256+q96 (aligned-tr.) |",
            "|---|---:|---:|---:|---:|"]
    for g in GENERATORS:
        rows.append(f"| {SHORT[g]} | " + " | ".join(f"{r[(g, 'mlp_clean+diff', c)]['balanced_accuracy']:.1%}"
                                                     for c in ("clean", "fmt_r256_q96") for r in (res_c, res_f)) + " |")
    lines += ["## mlp_clean+diff balanced accuracy per held-out generator", "\n".join(rows)]
    cpe.write_report("e2_format_aligned_training", lines, {f"{g}|{d}|{c}": v for (g, d, c), v in res_f.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["b1", "b2"])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    cpe.set_defaults(args)
    store = dh.EmbeddingStore(args)      # unlike cpe.make_store, this one may embed
    {"b1": run_b1, "b2": run_b2}[args.experiment](args, store)


if __name__ == "__main__":
    main()
