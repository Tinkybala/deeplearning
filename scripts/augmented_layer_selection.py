"""Augmented training for every DINOv2 layer probe, chosen by nested leave-one-generator-out.

The final detector (final_detector_experiments.py) is the block-20 MLP fit on clean +
augmented images. Augmented features exist for every DINOv2 block (augmented_backbone_
features.py), so this asks whether another block, the mean patch token or the multi-block
concatenation does better once it is also trained on augmented images, and chooses honestly.

Candidates: the 8 DINOv2 feature sets of backbone_comparison (blocks 8/12/16/20/24, block
16/24 mean patch token, blocks 12-24 concatenated) x {LR, MLP} x {clean, clean + augmented}
training = 32 single detectors, each with the clean seen-generator validation threshold (no
condition knowledge, single images).

Selection, as in nested_selection_experiments.py: for each outer held-out generator g and each
seen generator h, every candidate is refit on the other five seen generators and scored on h;
the inner score is the balanced accuracy on h averaged over the six standard conditions and
the six inner folds. The chosen candidate is refit on all six seen generators and tested on g.
About 1-1.5 h.

    python scripts/augmented_layer_selection.py
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
AUG = "clean__aug-both"
FEATURES = {k: v for k, v in bc.FEATURES.items() if "CLIP" not in k}
CANDS = [f"{f} {k} [{t}]" for f in FEATURES for k in ("LR", "MLP") for t in ("clean", "clean+both")]
FINAL = "DINOv2 L20 MLP [clean+both]"
STATE = ROOT / "results" / "_augmented_layer_selection_state.pkl"


def fold_scores(args, fstore, held_out: str, seen: list[str], real_source: str) -> tuple[dict, dict]:
    """Per candidate: {condition: BA on held_out} with the clean seen-val threshold, and the
    seen-val BA."""
    fit_rec, val_rec = cpe.splits(args, seen, real_source)
    val_new = bc.fa.shared_val(args, val_rec)
    test_rec = cpe.test_records(args, held_out)
    yf, yv = np.asarray([r[1] for r in fit_rec]), np.asarray([r[1] for r in val_rec])
    yt = np.asarray([r[1] for r in test_rec])
    out, val_ba = {}, {}
    for f, keys in FEATURES.items():
        Xc, Xa = fstore.gather(fit_rec, "clean", keys), fstore.gather(fit_rec, AUG, keys)
        Xv = fstore.gather(val_new, "clean", keys)
        tests = {c: fstore.gather(test_rec, c, keys) for c in CONDS}
        for k in ("LR", "MLP"):
            for t, (X, y) in (("clean", (Xc, yf)), ("clean+both", (np.vstack([Xc, Xa]), np.concatenate([yf, yf])))):
                fn = bc.fit_probe(k, X, y, args)
                thr, val_ba[f"{f} {k} [{t}]"] = cpe.best_threshold(yv, fn(Xv))
                out[f"{f} {k} [{t}]"] = {c: fn(tests[c]) for c in CONDS} | {"_thr": thr}
    return out, val_ba, yt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    if "--rebuild" in sys.argv[1:]:
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
        rules = {"nested pick (mean of 6 conditions)": lambda m: np.mean([inner[g][m][c] for c in STD]),
                 "nested pick (clean)": lambda m: inner[g][m]["clean"],
                 "seen-val pick": lambda m: seenval[g][m]}
        for rule, key in rules.items():
            m = max(CANDS, key=key)
            picks[(g, rule)] = m
            for c in CONDS:
                res[(g, rule, c)], boots[(g, rule, c)] = res[(g, m, c)], boots[(g, m, c)]
        print(f"{SHORT[g]:10s} nested pick {picks[(g, 'nested pick (mean of 6 conditions)')]!r}  "
              f"final {np.mean([res[(g, FINAL, c)]['balanced_accuracy'] for c in STD]):.3f}  "
              f"pick {np.mean([res[(g, 'nested pick (mean of 6 conditions)', c)]['balanced_accuracy'] for c in STD]):.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    STATE.write_bytes(pickle.dumps((res, boots, picks, inner, seenval)))
    report(args, res, boots, picks, inner, seenval)


def report(args, res, boots, picks, inner, seenval) -> None:
    gens = GENERATORS
    rules = ["nested pick (mean of 6 conditions)", "nested pick (clean)", "seen-val pick"]

    def mean(m, c):
        return float(np.mean([res[(g, m, c)]["balanced_accuracy"] for g in gens]))

    def m6(m):
        return float(np.mean([mean(m, c) for c in STD]))

    def inner6(m):
        return float(np.mean([np.mean([inner[g][m][c] for c in STD]) for g in gens]))

    lines = ["# Augmented training for every DINOv2 layer probe, with nested selection",
             "32 single detectors: 8 DINOv2 feature sets × {LR, MLP} × {clean, clean + augmented} training, each with "
             "the clean seen-generator validation threshold (single images, no condition knowledge). **Inner** = "
             "nested leave-one-generator-out balanced accuracy (refit on five seen generators, scored on the sixth), "
             "mean over the six standard conditions and the six inner folds, averaged over the 7 outer folds; it "
             "never uses the outer test generator. **Outer** = test balanced accuracy on the held-out generator. "
             f"**{FINAL}** is the final detector of final_detector_experiments.py. Mean over the 7 holdouts."]
    rows = ["| Candidate | Inner (6 cond.) | Seen val | " + " | ".join(LABEL[c] for c in CONDS) + " | Outer mean (6 std.) |",
            "|---|---:|---:|" + "---:|" * (len(CONDS) + 1)]
    for m in sorted(CANDS, key=lambda m: -inner6(m)):
        name = f"**{m}**" if m == FINAL else m
        rows.append(f"| {name} | {inner6(m):.1%} | {np.mean([seenval[g][m] for g in gens]):.1%} | "
                    + " | ".join(f"{mean(m, c):.1%}" for c in CONDS) + f" | {m6(m):.1%} |")
    lines += ["## All candidates, sorted by inner score", "\n".join(rows)]

    rows = ["| Selection rule | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |", "|---|" + "---:|" * (len(CONDS) + 1)]
    for m in [FINAL] + rules:
        rows.append(f"| {m} | " + " | ".join(f"{mean(m, c):.1%}" for c in CONDS) + f" | {m6(m):.1%} |")
    lines += ["## Selected detectors", "\n".join(rows)]

    rows = ["| Comparison | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |", "|---|" + "---:|" * (len(CONDS) + 1)]
    for a in rules:
        cells = []
        for c in CONDS + ["_mean"]:
            cs = STD if c == "_mean" else [c]
            point = np.mean([res[(g, a, x)]["balanced_accuracy"] - res[(g, FINAL, x)]["balanced_accuracy"] for g in gens for x in cs])
            rep = np.mean([boots[(g, a, x)]["balanced_accuracy"] - boots[(g, FINAL, x)]["balanced_accuracy"]
                           for g in gens for x in cs], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {a} − final | " + " | ".join(cells) + " |")
    lines += [f"## Difference from the final detector (pp), paired bootstrap ({args.bootstrap} replicates per holdout); "
              "* = 95% interval excludes 0", "\n".join(rows)]

    rows = ["| Held-out generator | " + " | ".join(rules) + " |", "|---|" + "---|" * len(rules)]
    rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, r)] for r in rules) + " |" for g in gens]
    counts = Counter(picks[(g, rules[0])] for g in gens)
    lines += ["## Picks per outer holdout", "\n".join(rows),
              "Nested picks (mean of 6 conditions): " + ", ".join(f"{m} ×{n}" for m, n in counts.most_common()) + "."]
    cpe.write_report("augmented_layer_selection", lines,
                     {f"{g}|{m}|{c}": v for (g, m, c), v in res.items()}
                     | {f"pick|{g}|{r}": v for (g, r), v in picks.items()}
                     | {f"inner|{g}": v for g, v in inner.items()} | {f"seenval|{g}": v for g, v in seenval.items()})


if __name__ == "__main__":
    main()
