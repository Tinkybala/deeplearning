"""Which frozen backbone and which layer? Probes on CLIP and on intermediate DINOv2 layers.

Uses the features from scripts/backbone_features.py (no backbone pass here) on the same
leave-one-generator-out protocol as the main runs: fit on the six seen generators' training
images, threshold from the clean seen-generator validation split, test on the held-out
generator under the six standard conditions and the format-matched condition fmt_r256_q96.

Feature sets (each probed with the main runs' logistic regression and MLP):
    DINOv2 L8 ... L24      CLS token after block 8/12/16/20/24 (L24 = the main runs' embedding)
    DINOv2 L16/L24 patch   mean patch token after block 16/24
    DINOv2 L12-24          CLS tokens of blocks 12, 16, 20, 24 concatenated
    CLIP                   OpenAI CLIP ViT-L/14 image embedding (the UnivFD backbone)
    DINOv2 L24 + CLIP      both final embeddings concatenated
Reference: the main runs' best detector, the MLP on DINOv2 clean + (noisy - clean).

Combinations:
    fusion   logit average of the reference MLP and the CLIP MLP
    router   per condition, the probe (any of the above, or fusion) and threshold that are best
             on the six seen generators' validation images degraded the same way (as E4)

Family holdout: for SD v1.5, Wukong, ADM and GLIDE the probes are also refit with the whole
family removed (as A1), giving a family-level clean number per feature set.

    python scripts/backbone_comparison.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_features as bf  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402
import format_alignment_experiments as fa  # noqa: E402
from train_final_probe import train_mlp  # noqa: E402

dh = fa.dh
GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS = dh.CONDITIONS + ["fmt_r256_q96"]
LABEL = fa.LABEL
FEATURES = {
    "DINOv2 L8": ["dino_cls8"], "DINOv2 L12": ["dino_cls12"], "DINOv2 L16": ["dino_cls16"],
    "DINOv2 L20": ["dino_cls20"], "DINOv2 L24": ["dino_cls24"],
    "DINOv2 L16 patch": ["dino_patch16"], "DINOv2 L24 patch": ["dino_patch24"],
    "DINOv2 L12-24": ["dino_cls12", "dino_cls16", "dino_cls20", "dino_cls24"],
    "CLIP": ["clip"], "DINOv2 L24 + CLIP": ["dino_cls24", "clip"],
}
REF = "DINOv2 clean+diff MLP (main runs)"
FUSION = "fusion: DINOv2 clean+diff MLP + CLIP MLP"


class FeatureStore:
    def __init__(self):
        self.memo = {}

    def folder(self, folder: Path, condition: str) -> dict:
        key = (folder, condition)
        if key not in self.memo:
            path = bf.cache_path(folder, condition)
            if not path.exists():
                raise SystemExit(f"Missing {path.relative_to(ROOT)}; run scripts/backbone_features.py first")
            data = dict(np.load(path))
            data["index"] = {n: i for i, n in enumerate(data["name"])}
            self.memo[key] = data
        return self.memo[key]

    def gather(self, records: list, condition: str, keys: list[str]) -> np.ndarray:
        out = np.empty((len(records), 0), np.float32)
        blocks = []
        for k in keys:
            col = None
            for i, (path, _, _) in enumerate(records):
                data = self.folder(Path(path).parent, condition)
                if col is None:
                    col = np.empty((len(records), data[k].shape[1]), np.float32)
                col[i] = data[k][data["index"][Path(path).name]]
            blocks.append(col)
        return np.hstack(blocks) if blocks else out


def fit_probe(kind: str, X: np.ndarray, y: np.ndarray, args):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    if kind == "LR":
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=5000).fit(Xs, y)
        col = list(clf.classes_).index(1)
        return lambda Z: clf.predict_proba(scaler.transform(Z))[:, col]
    net = train_mlp(Xs, y, args.hidden, seed=args.seed)

    def score(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(scaler.transform(Z), dtype=torch.float32))).squeeze(1).numpy()
    return score


def scores_for(args, fstore, dstore, held_out: str, seen: list[str], with_degraded: bool,
               real_source: str | None = None) -> dict:
    """{method: {"val": s, "test": {c: s}, "dval": {c: s}}} for every probe on one training set.

    Real photos come from real_source's train/nature (default seen[0], as in the main runs)."""
    fit_rec, val_rec = cpe.splits(args, seen, real_source or seen[0])
    val_new = fa.shared_val(args, val_rec)
    test_rec = cpe.test_records(args, held_out)
    dval_rec = [r for g in seen for r in cpe.test_records(args, g)]
    y_fit = np.asarray([r[1] for r in fit_rec])
    conds = CONDS if with_degraded else ["clean"]
    out = {}

    # reference: DINOv2 clean+diff MLP exactly as in the main runs (RIGID embedding cache)
    fit, val = dstore.gather(fit_rec, "clean", "none"), dstore.gather(val_rec, "clean", "none")
    fn = dh.fit_detector("mlp_clean+diff", fit, args, held_out)[0]
    out[REF] = {"val": fn(val), "test": {c: fn(dstore.gather(test_rec, c, "none")) for c in conds}}
    if with_degraded:
        out[REF]["dval"] = {c: fn(dstore.gather(dval_rec, c, "none")) for c in CONDS}

    for name, keys in FEATURES.items():
        X = fstore.gather(fit_rec, "clean", keys)
        Xv = fstore.gather(val_new, "clean", keys)
        tests = {c: fstore.gather(test_rec, c, keys) for c in conds}
        dvals = {c: fstore.gather(dval_rec, c, keys) for c in CONDS} if with_degraded else {}
        for kind in ("LR", "MLP"):
            fn = fit_probe(kind, X, y_fit, args)
            m = f"{name} {kind}"
            out[m] = {"val": fn(Xv), "test": {c: fn(t) for c, t in tests.items()}}
            if with_degraded:
                out[m]["dval"] = {c: fn(d) for c, d in dvals.items()}

    a, b = out[REF], out["CLIP MLP"]
    fuse = lambda s, t: 0.5 * (cpe.logit(s) + cpe.logit(t))  # noqa: E731
    out[FUSION] = {"val": fuse(a["val"], b["val"]), "test": {c: fuse(a["test"][c], b["test"][c]) for c in conds}}
    if with_degraded:
        out[FUSION]["dval"] = {c: fuse(a["dval"][c], b["dval"][c]) for c in CONDS}
    out["_y"] = {"val": np.asarray([r[1] for r in val_rec]), "test": np.asarray([r[1] for r in test_rec]),
                 "dval": np.asarray([r[1] for r in dval_rec])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    fstore, dstore = FeatureStore(), cpe.make_store(args)
    rng = np.random.default_rng(args.boot_seed)
    res, boots, picks, family, valba = {}, {}, {}, {}, {}
    family_of = {g: m for m in cpe.FAMILIES.values() for g in m}

    for g in GENERATORS:
        t0 = time.time()
        seen = [x for x in GENERATORS if x != g]
        S = scores_for(args, fstore, dstore, g, seen, with_degraded=True)
        y = S.pop("_y")
        methods = list(S)
        boot = {c: cpe.Boot(y["test"], args.bootstrap, rng) for c in CONDS}
        for m in methods:
            thr, valba[(g, m)] = cpe.best_threshold(y["val"], S[m]["val"])
            for c in CONDS:
                res[(g, m, c)] = cpe.metrics(y["test"], S[m]["test"][c], thr)
                boots[(g, m, c)] = boot[c](S[m]["test"][c], thr)
        for c in CONDS:                           # router over every method, on degraded seen-val
            recal = {m: cpe.best_threshold(y["dval"], S[m]["dval"][c]) for m in methods}
            best = max(methods, key=lambda m: recal[m][1])
            picks[(g, c)] = best
            res[(g, "router", c)] = cpe.metrics(y["test"], S[best]["test"][c], recal[best][0])
            boots[(g, "router", c)] = boot[c](S[best]["test"][c], recal[best][0])
        # reproduction: DINOv2 L24 LR is the main runs' logreg_clean
        saved = dh.json.loads((ROOT / "results" / f"dinov2_logreg_clean_holdout_{g}" / "summary.json").read_text())
        same = abs(saved["test_metrics"]["clean"]["balanced_accuracy"] - res[(g, "DINOv2 L24 LR", "clean")]["balanced_accuracy"])
        print(f"{SHORT[g]:10s} ref {res[(g, REF, 'clean')]['balanced_accuracy']:.3f}  CLIP MLP "
              f"{res[(g, 'CLIP MLP', 'clean')]['balanced_accuracy']:.3f}  router {res[(g, 'router', 'clean')]['balanced_accuracy']:.3f}"
              f"  L24 LR {'reproduces logreg_clean' if same < 1e-9 else f'DIFFERS by {same:.4f}'}  ({time.time() - t0:.0f}s)",
              flush=True)

        if len(family_of[g]) > 1:                 # family holdout (clean test only)
            fam = [x for x in GENERATORS if x not in family_of[g]]
            F = scores_for(args, fstore, dstore, g, fam, with_degraded=False)
            yf = F.pop("_y")
            for m in F:
                thr = cpe.best_threshold(yf["val"], F[m]["val"])[0]
                family[(g, m)] = cpe.metrics(yf["test"], F[m]["test"]["clean"], thr)

    methods = [m for m in dict.fromkeys(k[1] for k in res)]
    report(args, res, boots, picks, family, methods, family_of, valba)


def report(args, res, boots, picks, family, methods, family_of, valba) -> None:
    gens = GENERATORS
    std = dh.CONDITIONS

    def mean(m, c, metric="balanced_accuracy"):
        return float(np.mean([res[(g, m, c)][metric] for g in gens]))

    def fam_mean(m):          # family-level clean BA: family-out for multi-member families, else LOGO
        return float(np.mean([family[(g, m)]["balanced_accuracy"] if len(family_of[g]) > 1
                              else res[(g, m, "clean")]["balanced_accuracy"] for g in gens]))

    lines = ["# Backbone and layer comparison (frozen features)",
             "Leave-one-generator-out over the 7 GenImage generators, same split, threshold rule and test conditions "
             "as the main runs. **LR** = logistic regression (C=0.1, class-balanced), **MLP** = the main runs' MLP probe "
             "(256 hidden, seed 42); both on standardised features. **Family out** = clean balanced accuracy with the "
             "held-out generator's whole family removed from training (SD v1.5 + Wukong, ADM + GLIDE), 7-generator "
             "mean as in A1. **Format 256+q96** = both classes resized to 256 and saved as JPEG q96 (E1). The router "
             "chooses, per condition, among all rows above it using only the seen generators' degraded validation "
             "images. **Seen val** = balanced accuracy on the clean seen-generator validation split (available without "
             "any test data, so it can be used to choose a layer). Mean over the 7 held-out generators; best per column "
             "in bold."]
    rows = ["| Features | Probe | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) | Family out | AUROC clean | Seen val |",
            "|---|---|" + "---:|" * (len(CONDS) + 4)]
    shown = [m for m in methods if m != "router"] + ["router"]
    best = {c: max(mean(m, c) for m in shown) for c in CONDS}
    for m in shown:
        feat, probe = (m, "") if m in (REF, FUSION, "router") else m.rsplit(" ", 1)
        cells = [(f"**{mean(m, c):.1%}**" if np.isclose(mean(m, c), best[c]) else f"{mean(m, c):.1%}") for c in CONDS]
        fo = f"{fam_mean(m):.1%}" if m != "router" else ""
        auc = f"{mean(m, 'clean', 'auroc_real_positive'):.3f}" if m != "router" else ""
        sv = f"{np.mean([valba[(g, m)] for g in gens]):.1%}" if m != "router" else ""
        rows.append(f"| {feat} | {probe} | " + " | ".join(cells)
                    + f" | {np.mean([mean(m, c) for c in std]):.1%} | {fo} | {auc} | {sv} |")
    lines += ["## Balanced accuracy", "\n".join(rows)]

    rows = ["| Method − reference | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
    for m in ["DINOv2 L20 LR", "DINOv2 L20 MLP", "DINOv2 L16 MLP", "DINOv2 L12-24 LR", "DINOv2 L24 patch MLP",
              "CLIP LR", "CLIP MLP", "DINOv2 L24 + CLIP MLP", FUSION, "router"]:
        cells = []
        for c in CONDS:
            point = np.mean([res[(g, m, c)]["balanced_accuracy"] - res[(g, REF, c)]["balanced_accuracy"] for g in gens])
            rep = np.mean([boots[(g, m, c)]["balanced_accuracy"] - boots[(g, REF, c)]["balanced_accuracy"] for g in gens], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += [f"## Difference from the reference ({REF}), balanced accuracy (pp), paired bootstrap "
              f"({args.bootstrap} replicates per holdout); * = 95% interval excludes 0", "\n".join(rows)]

    cols = [REF, "DINOv2 L24 LR", "DINOv2 L16 MLP", "DINOv2 L20 MLP", "CLIP MLP", "DINOv2 L24 + CLIP MLP", FUSION, "router"]
    rows = ["| Held-out generator | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for g in gens:
        vals = [res[(g, m, "clean")]["balanced_accuracy"] for m in cols]
        rows.append(f"| {SHORT[g]} | " + " | ".join(f"**{v:.1%}**" if v == max(vals) else f"{v:.1%}" for v in vals) + " |")
    lines += ["## Clean balanced accuracy per held-out generator", "\n".join(rows)]

    rows = ["| Held-out generator | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, c)] for c in CONDS) + " |" for g in gens]
    lines += ["## Router choice (on degraded seen-generator validation)", "\n".join(rows)]
    cpe.write_report("backbone_comparison", lines,
                     {f"{g}|{m}|{c}": v for (g, m, c), v in res.items()}
                     | {f"family|{g}|{m}": v for (g, m), v in family.items()}
                     | {f"pick|{g}|{c}": v for (g, c), v in picks.items()}
                     | {f"seenval|{g}|{m}": v for (g, m), v in valba.items()})


if __name__ == "__main__":
    main()
