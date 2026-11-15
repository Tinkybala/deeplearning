"""Degradation-augmented DINOv2 probes on the full leave-one-generator-out protocol.

The main runs fit their probes on clean embeddings only and need the router (which must be
told, or must guess, the test degradation) to stay accurate under JPEG and resizing. Here the
same probes are fit on embeddings of the seen generators' training images after the ResNet
study's random artifacts, applied once per image before embedding (seeded, reproducible):

    jpeg     JPEG quality 30-95 with p = 0.5
    resize   downsample to 32/64/128 px and back to 256 with p = 0.5
    both     both of the above, independently (about a quarter of the images stay clean)
    clean+both   the clean and the "both" embeddings together (twice the fit set)

Everything else is the main runs' protocol: seed-42 split, threshold maximising balanced
accuracy on the clean seen-generator validation split (a single detector that is not told the
test condition), 500 + 500 test images per held-out generator under the six standard and two
format-matched conditions. Also reported: the same probes with the threshold re-picked on
degraded seen-generator validation ("+recal", which needs the condition), and an E4-style
router whose pool adds the augmented probes.

Missing augmented embeddings (GLIDE train/ai and VQDM train/nature; the other folders were
cached by the earlier GLIDE-holdout runs) are computed on first use, about 12k images.

    python scripts/augmented_probe_experiments.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cached_probe_experiments as cpe  # noqa: E402
import format_alignment_experiments as fa  # noqa: E402

dh = fa.dh
GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS = dh.CONDITIONS + list(fa.FORMAT)
STD = dh.CONDITIONS
LABEL = fa.LABEL
TRAINS = ["clean", "jpeg", "resize", "both", "clean+both"]
PROBES = ["logreg_clean", "logreg_clean+diff", "mlp_clean+diff"]
REF = "mlp_clean+diff [clean]"


def concat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in ("clean", "noisy", "label", "group", "path")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    store = dh.EmbeddingStore(args)                  # embeds what is missing (augmented fit sets)
    rng = np.random.default_rng(args.boot_seed)
    res, boots, picks, repro = {}, {}, {}, {}

    for g in GENERATORS:
        t0 = time.time()
        seen = [x for x in GENERATORS if x != g]
        fit_rec, val_rec = cpe.splits(args, seen, seen[0])
        fits = {t: store.gather(fit_rec, "clean", t if t != "clean" else "none") for t in TRAINS if "+" not in t}
        fits["clean+both"] = concat(fits["clean"], fits["both"])
        val = store.gather(val_rec, "clean", "none")
        dval_rec = [r for x in seen for r in cpe.test_records(args, x)]
        dval = {c: store.gather(dval_rec, c, "none") for c in CONDS}
        tests = {c: store.gather(cpe.test_records(args, g), c, "none") for c in CONDS}
        yv, yd, yt = (val["label"].astype(int), dval["clean"]["label"].astype(int), tests["clean"]["label"].astype(int))
        boot = {c: cpe.Boot(yt, args.bootstrap, rng) for c in CONDS}

        scores = {}                                  # method -> {"val", "dval": {c}, "test": {c}}
        for t in TRAINS:
            for p in PROBES:
                fn = dh.fit_detector(p, fits[t], args, g)[0]
                scores[f"{p} [{t}]"] = {"val": fn(val), "dval": {c: fn(dval[c]) for c in CONDS},
                                        "test": {c: fn(tests[c]) for c in CONDS}}
        fz = lambda a, b: 0.5 * (cpe.logit(a) + cpe.logit(b))  # noqa: E731  (E4's fusion)
        a, b = scores["mlp_clean+diff [clean]"], scores["logreg_clean [clean]"]
        scores["fusion [clean]"] = {"val": fz(a["val"], b["val"]), "dval": {c: fz(a["dval"][c], b["dval"][c]) for c in CONDS},
                                    "test": {c: fz(a["test"][c], b["test"][c]) for c in CONDS}}

        for m, s in scores.items():
            thr = cpe.best_threshold(yv, s["val"])[0]
            for c in CONDS:
                res[(g, m, c)] = cpe.metrics(yt, s["test"][c], thr)
                boots[(g, m, c)] = boot[c](s["test"][c], thr)
                rt = cpe.best_threshold(yd, s["dval"][c])[0]
                res[(g, m + " +recal", c)] = cpe.metrics(yt, s["test"][c], rt)
                boots[(g, m + " +recal", c)] = boot[c](s["test"][c], rt)
        pools = {"router (E4 pool)": ["mlp_clean+diff [clean]", "logreg_clean [clean]", "fusion [clean]"],
                 "router (+ augmented probes)": list(scores)}
        for name, pool in pools.items():
            for c in CONDS:
                recal = {m: cpe.best_threshold(yd, scores[m]["dval"][c]) for m in pool}
                best = max(pool, key=lambda m: recal[m][1])
                picks[(g, name, c)] = best
                res[(g, name, c)] = cpe.metrics(yt, scores[best]["test"][c], recal[best][0])
                boots[(g, name, c)] = boot[c](scores[best]["test"][c], recal[best][0])

        if g == "imagenet_glide":                    # the earlier GLIDE-only augmented runs
            for t in ("jpeg", "resize", "both"):
                saved = dh.json.loads((ROOT / "results" / f"dinov2_mlp_clean_diff_{t}_holdout_{g}" / "summary.json").read_text())
                repro[t] = max(abs(saved["test_metrics"][c]["balanced_accuracy"]
                                   - res[(g, f"mlp_clean+diff [{t}]", c)]["balanced_accuracy"]) for c in STD)
        print(f"{SHORT[g]:10s} MLP clean/JPEG50/r64: " + "  ".join(
            f"{t} {res[(g, f'mlp_clean+diff [{t}]', 'clean')]['balanced_accuracy']:.3f}/"
            f"{res[(g, f'mlp_clean+diff [{t}]', 'jpeg_q50')]['balanced_accuracy']:.3f}/"
            f"{res[(g, f'mlp_clean+diff [{t}]', 'resize_64')]['balanced_accuracy']:.3f}" for t in TRAINS)
            + (f"  | GLIDE reproduction max |diff| {max(repro.values()):.4f}" if repro else "")
            + f"  ({time.time() - t0:.0f}s, {store.embedded} images embedded so far)", flush=True)
    report(args, res, boots, picks, repro)


def report(args, res, boots, picks, repro) -> None:
    gens = GENERATORS

    def mean(m, c, metric="balanced_accuracy"):
        return float(np.mean([res[(g, m, c)][metric] for g in gens]))

    def table(methods, metric="balanced_accuracy", f=lambda v: f"{v:.1%}"):
        rows = ["| Method | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |",
                "|---|" + "---:|" * (len(CONDS) + 1)]
        best = {c: max(mean(m, c, metric) for m in methods) for c in CONDS}
        for m in methods:
            cells = [f"**{f(mean(m, c, metric))}**" if np.isclose(mean(m, c, metric), best[c]) else f(mean(m, c, metric))
                     for c in CONDS]
            rows.append(f"| {m} | " + " | ".join(cells) + f" | {f(np.mean([mean(m, c, metric) for c in STD]))} |")
        return "\n".join(rows)

    def diffs(pairs):
        rows = ["| Comparison | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |",
                "|---|" + "---:|" * (len(CONDS) + 1)]
        for a, b in pairs:
            cells = []
            for c in CONDS + ["_mean"]:
                cs = STD if c == "_mean" else [c]
                point = np.mean([res[(g, a, x)]["balanced_accuracy"] - res[(g, b, x)]["balanced_accuracy"] for g in gens for x in cs])
                rep = np.mean([boots[(g, a, x)]["balanced_accuracy"] - boots[(g, b, x)]["balanced_accuracy"]
                               for g in gens for x in cs], axis=0)
                cells.append(cpe.fmt_diff(point, rep, 100, 1))
            rows.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    single = [f"{p} [{t}]" for p in ("mlp_clean+diff", "logreg_clean") for t in TRAINS] + ["fusion [clean]"]
    lines = ["# Degradation-augmented DINOv2 probes (all 7 holdouts)",
             "Probes fit on DINOv2 embeddings of the seen generators' training images after random artifacts applied "
             "once per image: **jpeg** = JPEG quality 30–95 (p = 0.5); **resize** = downsample to 32/64/128 px and back "
             "to 256 (p = 0.5); **both** = both, independently; **clean+both** = clean and *both* embeddings together. "
             "**[clean]** = the main runs' probes. Thresholds from the clean seen-generator validation split, so every "
             "row except the routers and *+recal* is a single detector that is not told the test condition. "
             "**+recal** = threshold re-picked on the seen generators' validation images degraded like the test "
             "condition; **router** = per condition, the probe and threshold best on those images (E4), from the E4 "
             "pool (clean MLP, clean logistic regression, their fusion) or from every probe here. Mean over the 7 held-out "
             "generators; best per column in bold."]
    if repro:
        lines.append("Reproduction: on the GLIDE holdout the jpeg / resize / both MLPs match the earlier GLIDE-only "
                     f"augmented runs to within {max(repro.values()) * 100:.2f} pp on every standard condition.")
    lines += ["## Single detectors (clean seen-val threshold), balanced accuracy", table(single),
              "## AUROC", table([f"mlp_clean+diff [{t}]" for t in TRAINS], "auroc_real_positive", lambda v: f"{v:.3f}"),
              "## Condition-aware variants, balanced accuracy",
              table([f"mlp_clean+diff [{t}] +recal" for t in TRAINS] + ["router (E4 pool)", "router (+ augmented probes)"]),
              f"## Differences in balanced accuracy (pp), 7-generator mean, paired bootstrap ({args.bootstrap} replicates "
              "per holdout); * = 95% interval excludes 0",
              diffs([(f"mlp_clean+diff [{t}]", REF) for t in TRAINS[1:]]
                    + [("mlp_clean+diff [both]", "router (E4 pool)"), ("mlp_clean+diff [clean+both]", "router (E4 pool)"),
                       ("router (+ augmented probes)", "router (E4 pool)")])]
    rows = ["| Method | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
    for t in TRAINS:
        m = f"mlp_clean+diff [{t}]"
        rows.append(f"| {m} | " + " | ".join(f"{mean(m, c, 'real_recall'):.0%} / {mean(m, c, 'generated_recall'):.0%}"
                                             for c in CONDS) + " |")
    lines += ["## Class recall (7-generator mean): real / generated", "\n".join(rows)]
    for c in ("clean", "jpeg_q50", "resize_64"):
        ms = [f"mlp_clean+diff [{t}]" for t in TRAINS] + ["router (E4 pool)"]
        rows = ["| Held-out generator | " + " | ".join(ms) + " |", "|---|" + "---:|" * len(ms)]
        for g in gens:
            vals = [res[(g, m, c)]["balanced_accuracy"] for m in ms]
            rows.append(f"| {SHORT[g]} | " + " | ".join(f"**{v:.1%}**" if v == max(vals) else f"{v:.1%}" for v in vals) + " |")
        lines += [f"## {LABEL[c]} balanced accuracy per held-out generator", "\n".join(rows)]
    rows = ["| Held-out generator | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, 'router (+ augmented probes)', c)] for c in CONDS) + " |" for g in gens]
    lines += ["## Router (+ augmented probes) choice", "\n".join(rows)]
    cpe.write_report("augmented_probes", lines,
                     {f"{g}|{m}|{c}": v for (g, m, c), v in res.items()}
                     | {f"pick|{g}|{n}|{c}": v for (g, n, c), v in picks.items()} | {f"repro|{k}": v for k, v in repro.items()})


if __name__ == "__main__":
    main()
