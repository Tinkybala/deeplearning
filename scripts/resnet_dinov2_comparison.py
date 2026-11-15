"""Fine-tuned ResNet-18 (Tyrone's checkpoints) vs the frozen DINOv2 probe, and their combination.

Uses the per-image ResNet scores from scripts/resnet_checkpoint_scores.py (no network pass
here) and the cached DINOv2 embeddings, on the leave-one-generator-out protocol all runs
share (seed-42 split, threshold maximising balanced accuracy on the clean seen-generator
validation split, 500 real + 500 generated test images per held-out generator).

Methods
    ResNet-18 squash224    Tyrone's fine-tuned checkpoint, Resize((224, 224)) at test time
                           (his published numbers; reproduced here from his own thresholds)
    ResNet-18 crop         the same checkpoint, Resize(256) + CenterCrop(224) (DINOv2's view)
    DINOv2 MLP             the main runs' clean + (noisy - clean) MLP
    ensemble (crop / squash)  mean of the two models' logits, each standardised on the clean
                           seen-generator validation split
    router                 per condition, the method and threshold best on the six seen
                           generators' validation images degraded the same way (as E4)
With --backbones the 22 probes of scripts/backbone_comparison.py join the router's pool.

Conditions: the six standard ones plus the format-matched Format q96 and Format 256+q96 (E1).

    python scripts/resnet_dinov2_comparison.py [--backbones]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cached_probe_experiments as cpe  # noqa: E402
import format_alignment_experiments as fa  # noqa: E402
import resnet_checkpoint_scores as rcs  # noqa: E402

dh = fa.dh
GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS = dh.CONDITIONS + list(fa.FORMAT)
STD = dh.CONDITIONS
LABEL = fa.LABEL
RN_SQ, RN_CC, DINO = "ResNet-18 squash224", "ResNet-18 crop", "DINOv2 MLP"
ENS_CC, ENS_SQ = "ensemble: ResNet crop + DINOv2", "ensemble: ResNet squash + DINOv2"
CORE = [RN_SQ, RN_CC, DINO, ENS_CC, ENS_SQ]


class ResNetScores:
    def __init__(self):
        self.memo = {}

    def gather(self, records: list, condition: str, held_out: str, view: str) -> np.ndarray:
        gi, vi = GENERATORS.index(held_out), rcs.PREPROC.index(view)
        out = np.empty(len(records))
        for i, (path, _, _) in enumerate(records):
            key = (Path(path).parent, condition)
            if key not in self.memo:
                p = rcs.cache_path(*key)
                if not p.exists():
                    raise SystemExit(f"Missing {p.relative_to(ROOT)}; run scripts/resnet_checkpoint_scores.py first")
                d = np.load(p)
                assert list(d["generators"]) == GENERATORS and tuple(d["preproc"]) == rcs.PREPROC
                self.memo[key] = (d["score"].astype(np.float64), {n: j for j, n in enumerate(d["name"])})
            s, index = self.memo[key]
            out[i] = s[index[Path(path).name], gi, vi]
        return out


def zfuse(a: dict, b: dict) -> dict:
    """Average of two methods' scores after standardising each on its clean seen-val scores."""
    def z(m, s):
        return (s - m["val"].mean()) / m["val"].std()
    return {part: ({c: 0.5 * (z(a, a[part][c]) + z(b, b[part][c])) for c in a[part]} if part != "val"
                   else 0.5 * (z(a, a["val"]) + z(b, b["val"]))) for part in ("val", "test", "dval")}


def scores_for(args, rstore, dstore, held_out: str, backbones) -> dict:
    seen = [g for g in GENERATORS if g != held_out]
    fit_rec, val_rec = cpe.splits(args, seen, seen[0])
    val_new = fa.shared_val(args, val_rec)
    test_rec = cpe.test_records(args, held_out)
    dval_rec = [r for g in seen for r in cpe.test_records(args, g)]
    out = {}
    for name, view in ((RN_SQ, "squash224"), (RN_CC, "center_crop")):
        out[name] = {"val": rstore.gather(val_new, "clean", held_out, view),
                     "test": {c: rstore.gather(test_rec, c, held_out, view) for c in CONDS},
                     "dval": {c: rstore.gather(dval_rec, c, held_out, view) for c in CONDS}}
    fn = dh.fit_detector("mlp_clean+diff", dstore.gather(fit_rec, "clean", "none"), args, held_out)[0]
    lg = lambda r, c: cpe.logit(fn(dstore.gather(r, c, "none")))  # noqa: E731
    out[DINO] = {"val": lg(val_rec, "clean"), "test": {c: lg(test_rec, c) for c in CONDS},
                 "dval": {c: lg(dval_rec, c) for c in CONDS}}
    out[ENS_CC], out[ENS_SQ] = zfuse(out[RN_CC], out[DINO]), zfuse(out[RN_SQ], out[DINO])
    if backbones is not None:                      # the 22 frozen-feature probes, for the router pool
        extra = backbones.scores_for(args, backbones.FeatureStore(), dstore, held_out, seen, with_degraded=True)
        extra.pop("_y")
        extra.pop(backbones.REF)                   # = DINO
        for m, v in extra.items():                 # backbone_comparison scores dval on its CONDS only
            out[m] = {"val": v["val"], "test": v["test"], "dval": v["dval"]}
    out["_y"] = {k: np.asarray([r[1] for r in rec]) for k, rec in
                 (("val", val_rec), ("test", test_rec), ("dval", dval_rec))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbones", action="store_true", help="add the 22 backbone_comparison probes to the router pool")
    parser.add_argument("--tyrone-ref", default="origin/tyrone",
                        help="git ref holding Tyrone's published summaries (reproduction check; skipped if absent)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    backbones = None
    if args.backbones:
        import backbone_comparison as backbones
    rstore, dstore = ResNetScores(), cpe.make_store(args)
    rng = np.random.default_rng(args.boot_seed)
    res, boots, picks, valba, repro = {}, {}, {}, {}, {}

    for g in GENERATORS:
        t0 = time.time()
        S = scores_for(args, rstore, dstore, g, backbones)
        y = S.pop("_y")
        boot = {c: cpe.Boot(y["test"], args.bootstrap, rng) for c in CONDS}
        for m in CORE:                             # the extra probes only feed the router
            thr, valba[(g, m)] = cpe.best_threshold(y["val"], S[m]["val"])
            for c in CONDS:
                res[(g, m, c)] = cpe.metrics(y["test"], S[m]["test"][c], thr)
                boots[(g, m, c)] = boot[c](S[m]["test"][c], thr)
        for c in CONDS:
            cand = [m for m in S if c in S[m]["dval"]]
            recal = {m: cpe.best_threshold(y["dval"], S[m]["dval"][c]) for m in cand}
            best = max(cand, key=lambda m: recal[m][1])
            picks[(g, c)] = best
            res[(g, "router", c)] = cpe.metrics(y["test"], S[best]["test"][c], recal[best][0])
            boots[(g, "router", c)] = boot[c](S[best]["test"][c], recal[best][0])

        # reproduction of Tyrone's published numbers: his own threshold on the squash224 scores
        pub = published(args.tyrone_ref, g)
        if pub:
            t = float(cpe.logit(np.asarray([pub["threshold_selected_on_seen_validation"]]))[0])
            repro[g] = {c: (cpe.metrics(y["test"], S[RN_SQ]["test"][c], t)["balanced_accuracy"],
                            pub["test_metrics"][c]["balanced_accuracy"]) for c in STD}
        worst = max(abs(a - b) for a, b in repro[g].values()) if g in repro else float("nan")
        print(f"{SHORT[g]:10s} ResNet sq {res[(g, RN_SQ, 'clean')]['balanced_accuracy']:.3f}  crop "
              f"{res[(g, RN_CC, 'clean')]['balanced_accuracy']:.3f}  DINOv2 {res[(g, DINO, 'clean')]['balanced_accuracy']:.3f}"
              f"  ens {res[(g, ENS_CC, 'clean')]['balanced_accuracy']:.3f}  | vs Tyrone's BA max |diff| {worst:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    report(args, res, boots, picks, valba, repro)


def report(args, res, boots, picks, valba, repro) -> None:
    gens = GENERATORS
    name = "resnet_dinov2_comparison" + ("_with_backbones" if args.backbones else "")

    def mean(m, c, metric="balanced_accuracy"):
        return float(np.mean([res[(g, m, c)][metric] for g in gens]))

    shown = CORE + ["router"]
    lines = ["# Fine-tuned ResNet-18 vs frozen DINOv2, and their combination",
             "ResNet-18 = Tyrone's seven leave-one-generator-out checkpoints (ImageNet-pretrained, fine-tuned end to "
             "end on the six seen generators with Resize 256 + RandomResizedCrop 224, no degradation augmentation). "
             "**squash224** = Resize((224, 224)) at test time, as in his published results; **crop** = Resize 256 + "
             "CenterCrop 224, the view the DINOv2 probes see. **DINOv2 MLP** = the main runs' clean + diff MLP. "
             "**Ensemble** = mean of the two models' logits after standardising each on the clean seen-generator "
             "validation split. **Router** = per condition, the method and threshold best on the six seen "
             "generators' validation images degraded the same way"
             + (" (pool: the rows above plus the 22 frozen-feature probes of backbone_comparison)" if args.backbones
                else " (pool: the rows above)") + ". Thresholds of every other row come from the clean seen-generator "
             "validation split. Same split, test images and conditions throughout; mean over the 7 held-out "
             "generators, best per column in bold."]
    if repro:
        worst = max(abs(a - b) for g in repro for a, b in repro[g].values())
        lines.append(f"Reproduction: with Tyrone's own checkpoint thresholds, the squash224 scores computed here give "
                     f"his published balanced accuracy on every holdout and standard condition to within "
                     f"{worst * 100:.2f} pp (GPU vs Apple-silicon arithmetic).")

    for metric, title, f in (("balanced_accuracy", "Balanced accuracy", lambda v: f"{v:.1%}"),
                             ("auroc_real_positive", "AUROC", lambda v: f"{v:.3f}")):
        ms = shown if metric == "balanced_accuracy" else CORE
        rows = ["| Method | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |" + (" Seen val |" if metric == "balanced_accuracy" else ""),
                "|---|" + "---:|" * (len(CONDS) + 1 + (metric == "balanced_accuracy"))]
        best = {c: max(mean(m, c, metric) for m in ms) for c in CONDS}
        for m in ms:
            cells = [f"**{f(mean(m, c, metric))}**" if np.isclose(mean(m, c, metric), best[c]) else f(mean(m, c, metric))
                     for c in CONDS]
            row = f"| {m} | " + " | ".join(cells) + f" | {f(np.mean([mean(m, c, metric) for c in STD]))} |"
            if metric == "balanced_accuracy":
                row += f" {np.mean([valba[(g, m)] for g in gens]):.1%} |" if m != "router" else " |"
            rows.append(row)
        lines += [f"## {title}", "\n".join(rows)]

    rows = ["| Method | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
    for m in CORE:
        rows.append(f"| {m} | " + " | ".join(f"{res_mean(res, m, c, 'real_recall'):.0%} / "
                                             f"{res_mean(res, m, c, 'generated_recall'):.0%}" for c in CONDS) + " |")
    lines += ["## Class recall (7-generator mean): real / generated", "\n".join(rows)]

    comps = [(RN_CC, RN_SQ), (DINO, RN_SQ), (DINO, RN_CC), (ENS_CC, DINO), (ENS_CC, RN_CC), (ENS_SQ, DINO),
             ("router", DINO), ("router", RN_SQ)]
    rows = ["| Comparison | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
    for a, b in comps:
        cells = []
        for c in CONDS:
            point = np.mean([res[(g, a, c)]["balanced_accuracy"] - res[(g, b, c)]["balanced_accuracy"] for g in gens])
            rep = np.mean([boots[(g, a, c)]["balanced_accuracy"] - boots[(g, b, c)]["balanced_accuracy"] for g in gens], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += [f"## Differences in balanced accuracy (pp), 7-generator mean, paired bootstrap ({args.bootstrap} "
              "replicates per holdout); * = 95% interval excludes 0", "\n".join(rows)]

    rows = ["| Method | Format q96 − clean | Format 256+q96 − clean |", "|---|---:|---:|"]
    for m in CORE:
        cells = []
        for c in fa.FORMAT:
            point = mean(m, c) - mean(m, "clean")
            rep = np.mean([boots[(g, m, c)]["balanced_accuracy"] - boots[(g, m, "clean")]["balanced_accuracy"]
                           for g in gens], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["## Format dependence (E1): change in balanced accuracy when both classes share the real photos' "
              "format (pp)", "\n".join(rows)]

    for c in ("clean", "jpeg_q50"):
        rows = ["| Held-out generator | " + " | ".join(shown) + " |", "|---|" + "---:|" * len(shown)]
        for g in gens:
            vals = [res[(g, m, c)]["balanced_accuracy"] for m in shown]
            rows.append(f"| {SHORT[g]} | " + " | ".join(f"**{v:.1%}**" if v == max(vals) else f"{v:.1%}" for v in vals) + " |")
        lines += [f"## {LABEL[c]} balanced accuracy per held-out generator", "\n".join(rows)]

    rows = ["| Held-out generator | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, c)] for c in CONDS) + " |" for g in gens]
    lines += ["## Router choice (on degraded seen-generator validation)", "\n".join(rows)]
    cpe.write_report(name, lines,
                     {f"{g}|{m}|{c}": v for (g, m, c), v in res.items()}
                     | {f"pick|{g}|{c}": v for (g, c), v in picks.items()}
                     | {f"seenval|{g}|{m}": v for (g, m), v in valba.items()}
                     | {f"repro|{g}|{c}": v for g in repro for c, v in repro[g].items()})


def published(ref: str, g: str) -> dict | None:
    """Tyrone's summary.json for holdout g, read from git without checking anything out."""
    import subprocess
    out = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:results/{rcs.run_name(g)}/summary.json"],
                         capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


def res_mean(res, m, c, metric):
    return float(np.mean([res[(g, m, c)][metric] for g in GENERATORS]))


if __name__ == "__main__":
    main()
