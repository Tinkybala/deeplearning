"""Nested leave-one-generator-out model selection.

Every choice in backbone_comparison.py (which layer, LR or MLP, CLIP fusion, the router's
pick) was judged either after the fact on the held-out test generator, or on the seen
generators' own validation images, which rewards fitting the seen generators (blocks 12-16
reach 95-97% there and collapse on unseen generators). Here the choice is made the way the
deployment problem is posed: by generalisation to a generator the probe has not seen.

For each outer held-out generator g (the test set, never touched by selection) and each of
the six seen generators h, every candidate (the 22 frozen-feature probes, the main runs'
DINOv2 clean+diff MLP and its CLIP fusion) is refit on the other five seen generators and
scored on h's test images. A candidate's inner score is its balanced accuracy on h,
averaged over the six inner folds. The real photos are the outer fold's (seen[0]'s
train/nature) in every inner fold. Selection rules:

    seen-val pick      highest clean seen-generator validation BA (the naive rule)
    nested clean pick  highest inner clean BA (clean seen-val threshold in each fold)
    nested one detector  highest inner BA averaged over the six standard conditions, clean
                       seen-val threshold (one detector for every condition)
    router             E4/backbone router: per condition, the candidate and threshold best on
                       the seen generators' validation images degraded the same way
    nested router      per condition, the candidate with the highest inner BA under that
                       condition (threshold re-picked on the inner folds' degraded seen-val),
                       deployed with the threshold re-picked on the outer degraded seen-val

The chosen candidate is then refit on all six seen generators (= the backbone_comparison
fit) and scored on g. About 40 min (49 refits of 24 candidates).

    python scripts/nested_selection_experiments.py
"""

from __future__ import annotations

import argparse
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

dh = bc.dh
GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS, STD, LABEL = bc.CONDS, bc.dh.CONDITIONS, bc.LABEL
RULES = ["seen-val pick", "nested clean pick", "nested one detector", "router", "nested router"]


def ba_tables(S: dict, y: dict) -> tuple[dict, dict, dict]:
    """Per method: clean seen-val BA, test BA per condition with the clean-val threshold, and
    with the threshold re-picked on degraded seen-val."""
    val, clean_thr, recal_thr = {}, {}, {}
    for m, s in S.items():
        t, val[m] = cpe.best_threshold(y["val"], s["val"])
        clean_thr[m] = {c: cpe.metrics(y["test"], s["test"][c], t)["balanced_accuracy"] for c in CONDS}
        recal_thr[m] = {c: cpe.metrics(y["test"], s["test"][c], cpe.best_threshold(y["dval"], s["dval"][c])[0])
                        ["balanced_accuracy"] for c in CONDS}
    return val, clean_thr, recal_thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    fstore, dstore = bc.FeatureStore(), cpe.make_store(args)
    rng = np.random.default_rng(args.boot_seed)
    res, boots, picks, inner_log = {}, {}, {}, {}

    for g in GENERATORS:
        t0 = time.time()
        seen = [x for x in GENERATORS if x != g]
        S = bc.scores_for(args, fstore, dstore, g, seen, with_degraded=True)
        y = S.pop("_y")
        methods = list(S)
        inner_clean, inner_recal = {m: [] for m in methods}, {m: [] for m in methods}
        for h in seen:                                    # inner folds: h plays the unseen generator
            Si = bc.scores_for(args, fstore, dstore, h, [x for x in seen if x != h], with_degraded=True,
                               real_source=seen[0])
            yi = Si.pop("_y")
            _, cl, rc = ba_tables(Si, yi)
            for m in methods:
                inner_clean[m].append(cl[m])
                inner_recal[m].append(rc[m])
        icl = {m: {c: float(np.mean([f[c] for f in inner_clean[m]])) for c in CONDS} for m in methods}
        irc = {m: {c: float(np.mean([f[c] for f in inner_recal[m]])) for c in CONDS} for m in methods}
        inner_log[g] = {"clean_threshold": icl, "recal_threshold": irc,
                        "outer_clean_threshold": ba_tables(S, y)[1]}

        val_ba = {m: cpe.best_threshold(y["val"], S[m]["val"])[1] for m in methods}
        choice = {"seen-val pick": {c: max(methods, key=val_ba.get) for c in CONDS},
                  "nested clean pick": {c: max(methods, key=lambda m: icl[m]["clean"]) for c in CONDS},
                  "nested one detector": {c: max(methods, key=lambda m: np.mean([icl[m][x] for x in STD])) for c in CONDS}}
        recal = {c: {m: cpe.best_threshold(y["dval"], S[m]["dval"][c]) for m in methods} for c in CONDS}
        choice["router"] = {c: max(methods, key=lambda m: recal[c][m][1]) for c in CONDS}
        choice["nested router"] = {c: max(methods, key=lambda m: irc[m][c]) for c in CONDS}

        boot = {c: cpe.Boot(y["test"], args.bootstrap, rng) for c in CONDS}
        for m in [bc.REF, "DINOv2 L20 MLP", bc.FUSION]:     # fixed references
            thr = cpe.best_threshold(y["val"], S[m]["val"])[0]
            for c in CONDS:
                res[(g, m, c)] = cpe.metrics(y["test"], S[m]["test"][c], thr)
                boots[(g, m, c)] = boot[c](S[m]["test"][c], thr)
        for rule in RULES:
            for c in CONDS:
                m = choice[rule][c]
                thr = recal[c][m][0] if "router" in rule else cpe.best_threshold(y["val"], S[m]["val"])[0]
                picks[(g, rule, c)] = m
                res[(g, rule, c)] = cpe.metrics(y["test"], S[m]["test"][c], thr)
                boots[(g, rule, c)] = boot[c](S[m]["test"][c], thr)
        print(f"{SHORT[g]:10s} picks: seen-val {choice['seen-val pick']['clean']!r}, nested clean "
              f"{choice['nested clean pick']['clean']!r}, one detector {choice['nested one detector']['clean']!r}; "
              f"clean BA ref {res[(g, bc.REF, 'clean')]['balanced_accuracy']:.3f} nested-clean "
              f"{res[(g, 'nested clean pick', 'clean')]['balanced_accuracy']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    import pickle                                     # lets the report be rebuilt without refitting
    (ROOT / "results" / "_nested_selection_state.pkl").write_bytes(pickle.dumps((res, boots, picks, inner_log)))
    report(args, res, boots, picks, inner_log)


def report(args, res, boots, picks, inner_log) -> None:
    gens = GENERATORS

    def mean(m, c):
        return float(np.mean([res[(g, m, c)]["balanced_accuracy"] for g in gens]))

    shown = [bc.REF, "DINOv2 L20 MLP", bc.FUSION] + RULES
    lines = ["# Nested leave-one-generator-out model selection",
             "Candidates: the 22 frozen-feature probes of backbone_comparison, the main runs' DINOv2 clean+diff MLP "
             "(reference) and its fusion with the CLIP MLP. **Inner score** = balanced accuracy on a seen generator "
             "held out from a refit on the other five, averaged over the six inner folds; the outer held-out "
             "generator is never used for selection. **seen-val pick** = best clean seen-generator validation BA; "
             "**nested clean pick** = best inner clean BA; **nested one detector** = best inner BA averaged over the "
             "six standard conditions (one probe and its clean seen-val threshold for every condition); **router** = "
             "per condition, best on the seen generators' degraded validation images (as before); **nested router** = "
             "per condition, best inner BA under that condition, with the threshold re-picked on degraded seen-val. "
             "*DINOv2 L20 MLP* and *fusion* are the post-hoc best rows of backbone_comparison, shown for reference. "
             "Mean over the 7 held-out generators; best per column in bold."]
    rows = ["| Method | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |", "|---|" + "---:|" * (len(CONDS) + 1)]
    best = {c: max(mean(m, c) for m in shown) for c in CONDS}
    for m in shown:
        cells = [f"**{mean(m, c):.1%}**" if np.isclose(mean(m, c), best[c]) else f"{mean(m, c):.1%}" for c in CONDS]
        rows.append(f"| {m} | " + " | ".join(cells) + f" | {np.mean([mean(m, c) for c in STD]):.1%} |")
    lines += ["## Balanced accuracy", "\n".join(rows)]

    rows = ["| Comparison | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |", "|---|" + "---:|" * (len(CONDS) + 1)]
    for a, b in [("seen-val pick", bc.REF), ("nested clean pick", bc.REF), ("nested one detector", bc.REF),
                 ("nested router", bc.REF), ("nested router", "router"), ("nested clean pick", "seen-val pick")]:
        cells = []
        for c in CONDS + ["_mean"]:
            cs = STD if c == "_mean" else [c]
            point = np.mean([res[(g, a, x)]["balanced_accuracy"] - res[(g, b, x)]["balanced_accuracy"] for g in gens for x in cs])
            rep = np.mean([boots[(g, a, x)]["balanced_accuracy"] - boots[(g, b, x)]["balanced_accuracy"]
                           for g in gens for x in cs], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += [f"## Differences in balanced accuracy (pp), paired bootstrap ({args.bootstrap} replicates per holdout); "
              "* = 95% interval excludes 0", "\n".join(rows)]

    rows = ["| Held-out generator | seen-val pick | nested clean pick | nested one detector |", "|---|---|---|---|"]
    rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, r, 'clean')] for r in RULES[:3]) + " |" for g in gens]
    lines += ["## Single-detector choices per outer holdout", "\n".join(rows)]
    for rule in ("router", "nested router"):
        rows = ["| Held-out generator | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---|" * len(CONDS)]
        rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, rule, c)] for c in CONDS) + " |" for g in gens]
        lines += [f"## {rule} choices", "\n".join(rows)]

    # how well does each rule's signal rank the candidates? inner (nested) vs seen-val, against outer test BA
    cands = list(inner_log[gens[0]]["clean_threshold"])
    rows = ["| Candidate | Inner clean BA | Inner mean (6 std.) | Outer clean BA |", "|---|---:|---:|---:|"]
    top = sorted(cands, key=lambda m: -np.mean([inner_log[g]["clean_threshold"][m]["clean"] for g in gens]))
    for m in top[:10]:
        ic = np.mean([inner_log[g]["clean_threshold"][m]["clean"] for g in gens])
        im = np.mean([np.mean([inner_log[g]["clean_threshold"][m][c] for c in STD]) for g in gens])
        oc = np.mean([inner_log[g]["outer_clean_threshold"][m]["clean"] for g in gens])
        rows.append(f"| {m} | {ic:.1%} | {im:.1%} | {oc:.1%} |")
    lines += ["## Top 10 candidates by inner clean BA (7-holdout mean)", "\n".join(rows)]
    counts = Counter(picks[(g, "nested one detector", "clean")] for g in gens)
    lines.append("Nested one-detector picks across the 7 outer folds: "
                 + ", ".join(f"{m} ×{n}" for m, n in counts.most_common()) + ".")
    cpe.write_report("nested_selection", lines,
                     {f"{g}|{m}|{c}": v for (g, m, c), v in res.items()}
                     | {f"pick|{g}|{r}|{c}": v for (g, r, c), v in picks.items()}
                     | {f"inner|{g}": v for g, v in inner_log.items()})


if __name__ == "__main__":
    main()
