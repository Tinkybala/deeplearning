"""Intermediate CLIP blocks, alone and combined with the DINOv2 final detector; nested selection.

Uses the CLIP block features of clip_layer_features.py and the DINOv2 block features of
backbone_features.py / augmented_backbone_features.py. Candidates (single detectors, clean
seen-generator validation threshold, single images, no condition knowledge):

    CLIP Lk {LR, MLP} [clean+both]     k = 8, 12, 16, 20, 24 (residual-stream CLS after block k)
    CLIP Lk MLP [clean]                 k = 12, 16, 20, 24 (does augmentation matter for CLIP too?)
    DINOv2 L20 MLP [clean+both]         the final detector
    DINOv2 L12-24 LR [clean+both]       the best augmented DINOv2 probe (augmented_layer_selection)
    fusion DINOv2 L20 + CLIP Lk         mean of the two MLPs' logits, each standardised on clean seen-val
    DINOv2 L20 + CLIP Lk MLP            one MLP on both feature vectors, k = 12, 16, 20

Selection is nested leave-one-generator-out (as augmented_layer_selection.py): inner score =
balanced accuracy on a seen generator held out from a refit, averaged over the six standard
conditions and the six inner folds; the chosen candidate is refit on all six seen generators
and tested on the outer held-out generator. About 1-1.5 h.

    python scripts/clip_layer_experiments.py [--rebuild]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_comparison as bc  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402

GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS, STD, LABEL = bc.CONDS, bc.dh.CONDITIONS, bc.LABEL
AUG, CLIPS = "clean__aug-both", "__clipmid"
KS = [8, 12, 16, 20, 24]
FINAL = "DINOv2 L20 MLP [clean+both]"
STATE = ROOT / "results" / "_clip_layer_experiments_state.pkl"

# name -> (list of (feature source, keys), probe kind, training set)
SPECS = {FINAL: ([("dino", ["dino_cls20"])], "MLP", "clean+both"),
         "DINOv2 L12-24 LR [clean+both]": ([("dino", ["dino_cls12", "dino_cls16", "dino_cls20", "dino_cls24"])], "LR", "clean+both")}
for k in KS:
    for kind in ("LR", "MLP"):
        SPECS[f"CLIP L{k} {kind} [clean+both]"] = ([("clip", [f"clip_cls{k}"])], kind, "clean+both")
for k in KS[1:]:
    SPECS[f"CLIP L{k} MLP [clean]"] = ([("clip", [f"clip_cls{k}"])], "MLP", "clean")
for k in (12, 16, 20):
    SPECS[f"DINOv2 L20 + CLIP L{k} MLP [clean+both]"] = ([("dino", ["dino_cls20"]), ("clip", [f"clip_cls{k}"])], "MLP", "clean+both")
FUSIONS = {f"fusion DINOv2 L20 + CLIP L{k} [clean+both]": (FINAL, f"CLIP L{k} MLP [clean+both]") for k in KS}
CANDS = list(SPECS) + list(FUSIONS)


def gather(fstore, records, condition, parts):
    blocks = []
    for src, keys in parts:
        blocks.append(fstore.gather(records, condition + (CLIPS if src == "clip" else ""), keys))
    return np.hstack(blocks)


def fold_scores(args, fstore, held_out: str, seen: list[str], real_source: str):
    fit_rec, val_rec = cpe.splits(args, seen, real_source)
    val_new = bc.fa.shared_val(args, val_rec)
    test_rec = cpe.test_records(args, held_out)
    yf, yv = np.asarray([r[1] for r in fit_rec]), np.asarray([r[1] for r in val_rec])
    out, val_ba = {}, {}
    for name, (parts, kind, train) in SPECS.items():
        Xc = gather(fstore, fit_rec, "clean", parts)
        X, y = (Xc, yf) if train == "clean" else (np.vstack([Xc, gather(fstore, fit_rec, AUG, parts)]), np.concatenate([yf, yf]))
        fn = bc.fit_probe(kind, X, y, args)
        sv = fn(gather(fstore, val_new, "clean", parts))
        out[name] = {"val": sv} | {c: fn(gather(fstore, test_rec, c, parts)) for c in CONDS}
    for name, (a, b) in FUSIONS.items():
        za, zb = cpe.logit(out[a]["val"]), cpe.logit(out[b]["val"])
        f = lambda s, z: (cpe.logit(s) - z.mean()) / z.std()  # noqa: E731
        out[name] = {k: 0.5 * (f(out[a][k], za) + f(out[b][k], zb)) for k in ["val"] + CONDS}
    for name in CANDS:
        out[name]["_thr"], val_ba[name] = cpe.best_threshold(yv, out[name]["val"])
    return out, val_ba, np.asarray([r[1] for r in test_rec])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    if args.rebuild:
        return report(args, *pickle.loads(STATE.read_bytes()))
    fstore = bc.FeatureStore()
    rng = np.random.default_rng(args.boot_seed)
    res, boots, picks, inner, seenval = {}, {}, {}, {}, {}
    for g in GENERATORS:
        t0 = time.time()
        seen = [x for x in GENERATORS if x != g]
        S, seenval[g], yt = fold_scores(args, fstore, g, seen, seen[0])
        acc = {m: [] for m in CANDS}
        for h in seen:
            Si, _, yi = fold_scores(args, fstore, h, [x for x in seen if x != h], seen[0])
            for m in CANDS:
                acc[m].append({c: cpe.metrics(yi, Si[m][c], Si[m]["_thr"])["balanced_accuracy"] for c in CONDS})
        inner[g] = {m: {c: float(np.mean([a[c] for a in acc[m]])) for c in CONDS} for m in CANDS}
        boot = {c: cpe.Boot(yt, args.bootstrap, rng) for c in CONDS}
        for m in CANDS:
            for c in CONDS:
                res[(g, m, c)] = cpe.metrics(yt, S[m][c], S[m]["_thr"])
                boots[(g, m, c)] = boot[c](S[m][c], S[m]["_thr"])
        for rule, key in {"nested pick (mean of 6 conditions)": lambda m: np.mean([inner[g][m][c] for c in STD]),
                          "nested pick (clean)": lambda m: inner[g][m]["clean"]}.items():
            m = max(CANDS, key=key)
            picks[(g, rule)] = m
            for c in CONDS:
                res[(g, rule, c)], boots[(g, rule, c)] = res[(g, m, c)], boots[(g, m, c)]
        print(f"{SHORT[g]:10s} nested pick {picks[(g, 'nested pick (mean of 6 conditions)')]!r}: "
              f"{np.mean([res[(g, 'nested pick (mean of 6 conditions)', c)]['balanced_accuracy'] for c in STD]):.3f} "
              f"vs final {np.mean([res[(g, FINAL, c)]['balanced_accuracy'] for c in STD]):.3f}  ({time.time() - t0:.0f}s)", flush=True)
        STATE.write_bytes(pickle.dumps((res, boots, picks, inner, seenval)))
    report(args, res, boots, picks, inner, seenval)


def report(args, res, boots, picks, inner, seenval) -> None:
    gens = [g for g in GENERATORS if (g, FINAL, "clean") in res]
    rules = ["nested pick (mean of 6 conditions)", "nested pick (clean)"]

    def mean(m, c):
        return float(np.mean([res[(g, m, c)]["balanced_accuracy"] for g in gens]))

    def inner6(m):
        return float(np.mean([np.mean([inner[g][m][c] for c in STD]) for g in gens]))

    lines = ["# Intermediate CLIP blocks, and CLIP + DINOv2, with nested selection",
             f"Single detectors with the clean seen-generator validation threshold. **[clean+both]** = fit on clean + "
             f"augmented images (JPEG 30–95 and/or resize to 32/64/128 px, p = 0.5 each). **CLIP Lk** = CLS token of the "
             f"residual stream after block k of OpenAI CLIP ViT-L/14. **fusion** = mean of the DINOv2 block-20 MLP's and "
             f"the CLIP MLP's logits, each standardised on clean seen-val. **Inner** = nested leave-one-generator-out "
             f"balanced accuracy, mean over the six standard conditions (never uses the outer test generator). Mean over "
             f"{len(gens)} holdouts; **{FINAL}** is the final detector."]
    rows = ["| Candidate | Inner (6 cond.) | Seen val | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |",
            "|---|---:|---:|" + "---:|" * (len(CONDS) + 1)]
    for m in sorted(CANDS, key=lambda m: -inner6(m)):
        rows.append(f"| {'**' + m + '**' if m == FINAL else m} | {inner6(m):.1%} | {np.mean([seenval[g][m] for g in gens]):.1%} | "
                    + " | ".join(f"{mean(m, c):.1%}" for c in CONDS) + f" | {np.mean([mean(m, c) for c in STD]):.1%} |")
    lines += ["## All candidates, sorted by inner score", "\n".join(rows)]

    rows = ["| Comparison | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |", "|---|" + "---:|" * (len(CONDS) + 1)]
    others = rules + sorted([m for m in CANDS if m != FINAL], key=lambda m: -inner6(m))[:4]
    for a in others:
        cells = []
        for c in CONDS + ["_mean"]:
            cs = STD if c == "_mean" else [c]
            point = np.mean([res[(g, a, x)]["balanced_accuracy"] - res[(g, FINAL, x)]["balanced_accuracy"] for g in gens for x in cs])
            rep = np.mean([boots[(g, a, x)]["balanced_accuracy"] - boots[(g, FINAL, x)]["balanced_accuracy"]
                           for g in gens for x in cs], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {a} − final | " + " | ".join(cells) + " |")
    lines += [f"## Difference from the final detector (pp), paired bootstrap ({args.bootstrap} replicates per holdout); "
              "* = 95% interval excludes 0. Rows below the two rules are the top candidates by inner score.", "\n".join(rows)]

    rows = ["| Held-out generator | " + " | ".join(rules) + " |", "|---|" + "---|" * len(rules)]
    rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, r)] for r in rules) + " |" for g in gens]
    lines += ["## Picks per outer holdout", "\n".join(rows),
              "Nested picks (mean of 6 conditions): " + ", ".join(
                  f"{m} ×{n}" for m, n in Counter(picks[(g, rules[0])] for g in gens).most_common()) + "."]
    for c in ("clean", "jpeg_q50"):
        ms = [FINAL] + sorted([m for m in CANDS if m != FINAL], key=lambda m: -inner6(m))[:4]
        rows = ["| Held-out generator | " + " | ".join(ms) + " |", "|---|" + "---:|" * len(ms)]
        for g in gens:
            vals = [res[(g, m, c)]["balanced_accuracy"] for m in ms]
            rows.append(f"| {SHORT[g]} | " + " | ".join(f"**{v:.1%}**" if v == max(vals) else f"{v:.1%}" for v in vals) + " |")
        lines += [f"## {LABEL[c]} balanced accuracy per held-out generator", "\n".join(rows)]
    cpe.write_report("clip_layer_experiments", lines,
                     {f"{g}|{m}|{c}": v for (g, m, c), v in res.items()}
                     | {f"pick|{g}|{r}": v for (g, r), v in picks.items()}
                     | {f"inner|{g}": v for g, v in inner.items()} | {f"seenval|{g}": v for g, v in seenval.items()})


if __name__ == "__main__":
    main()
