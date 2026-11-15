"""How much of the accuracy lost under degradation is only a threshold shift?

Under JPEG and downsampling, AUROC falls much less than balanced accuracy (ResNet-18 at 128
px: AUROC 0.80, BA 66%; at 64 px it calls almost everything generated): the scores still
rank real above generated, but the threshold from clean seen-generator validation no longer
sits between them. This compares threshold rules for fixed detectors:

    clean val     the main runs' rule: threshold maximising BA on clean seen-generator val
    degraded val  re-picked on the seen generators' validation images degraded like the test
                  condition (needs the condition and labelled degraded data; E4's "+recal")
    test median   label-free: the median score of the unlabelled test batch (assumes half of
                  the batch is real)
    test GMM      label-free: a two-component Gaussian mixture on the batch's logit scores;
                  threshold where the two component densities cross (BA-optimal under the
                  mixture, no prior assumed)
    oracle        the BA-optimal threshold on the test labels (upper bound; not a method)

Label-free rules see one batch of unlabelled test images from one held-out generator and one
condition. Because the test batch is 50% real, the median rule gets the prior right by
construction, so every rule is also scored on batches with 20% and 80% real images (all of
the majority class plus a random subset of the minority class, 20 draws).

Detectors: DINOv2 clean+diff MLP (main runs), DINOv2 L20 MLP, DINOv2 + CLIP fusion, DINOv2
L24 logistic regression, ResNet-18 squash224 (Tyrone's checkpoints). Needs the caches of
backbone_features.py and resnet_checkpoint_scores.py.

    python scripts/threshold_adaptation_experiments.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_comparison as bc  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402
import resnet_dinov2_comparison as rd  # noqa: E402

GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS, STD, LABEL = bc.CONDS, bc.dh.CONDITIONS, bc.LABEL
DETECTORS = [rd.DINO, "DINOv2 L20 MLP", bc.FUSION, "DINOv2 L24 LR", rd.RN_SQ]
RULES = ["clean val", "degraded val", "test median", "test GMM", "oracle"]
PRIORS = [0.5, 0.2, 0.8]


def to_logit(s: np.ndarray, is_prob: bool) -> np.ndarray:
    return cpe.logit(s) if is_prob else s.astype(np.float64)


def gmm_threshold(s: np.ndarray) -> float:
    gm = GaussianMixture(2, random_state=0, reg_covar=1e-4).fit(s.reshape(-1, 1))
    mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
    lo, hi = np.argsort(mu)
    if mu[hi] - mu[lo] < 1e-6:
        return float(np.median(s))
    x = np.linspace(mu[lo], mu[hi], 2001)
    logpdf = lambda m, v: -np.log(v) - 0.5 * ((x - m) / v) ** 2  # noqa: E731
    return float(x[np.argmin(np.abs(logpdf(mu[lo], sd[lo]) - logpdf(mu[hi], sd[hi])))])


def thresholds(rule: str, s: dict, c: str, test: np.ndarray, y_test: np.ndarray, y: dict) -> float:
    if rule == "clean val":
        return cpe.best_threshold(y["val"], s["val"])[0]
    if rule == "degraded val":
        return cpe.best_threshold(y["dval"], s["dval"][c])[0]
    if rule == "test median":
        return float(np.median(test))
    if rule == "test GMM":
        return gmm_threshold(test)
    return cpe.best_threshold(y_test, test)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=20, help="random minority subsets per skewed prior")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    rstore, dstore = rd.ResNetScores(), cpe.make_store(args)
    rng, sub_rng = np.random.default_rng(args.boot_seed), np.random.default_rng(1)
    res, boots, skew = {}, {}, {}

    for g in GENERATORS:
        t0 = time.time()
        S = rd.scores_for(args, rstore, dstore, g, bc)
        y = S.pop("_y")
        yt = y["test"]
        boot = {c: cpe.Boot(yt, args.bootstrap, rng) for c in CONDS}
        real, fake = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
        for d in DETECTORS:
            is_prob = S[d]["val"].min() >= 0 and S[d]["val"].max() <= 1
            s = {"val": to_logit(S[d]["val"], is_prob), "dval": {c: to_logit(S[d]["dval"][c], is_prob) for c in CONDS},
                 "test": {c: to_logit(S[d]["test"][c], is_prob) for c in CONDS}}
            for c in CONDS:
                t_all = s["test"][c]
                for rule in RULES:
                    thr = thresholds(rule, s, c, t_all, yt, y)
                    res[(g, d, rule, c)] = cpe.metrics(yt, t_all, thr)
                    boots[(g, d, rule, c)] = boot[c](t_all, thr)
                for p in PRIORS[1:]:                      # skewed batches
                    k = round(len(fake) * p / (1 - p)) if p < 0.5 else round(len(real) * (1 - p) / p)
                    acc = {rule: [] for rule in RULES}
                    for _ in range(args.draws):
                        idx = (np.concatenate([sub_rng.choice(real, k, replace=False), fake]) if p < 0.5
                               else np.concatenate([real, sub_rng.choice(fake, k, replace=False)]))
                        for rule in RULES:
                            thr = thresholds(rule, s, c, t_all[idx], yt[idx], y)
                            acc[rule].append(cpe.metrics(yt[idx], t_all[idx], thr)["balanced_accuracy"])
                    for rule in RULES:
                        skew[(g, d, rule, c, p)] = float(np.mean(acc[rule]))
        print(f"{SHORT[g]:10s} DINOv2 MLP JPEG50 BA: " + "  ".join(
            f"{r} {res[(g, rd.DINO, r, 'jpeg_q50')]['balanced_accuracy']:.3f}" for r in RULES)
            + f"  ({time.time() - t0:.0f}s)", flush=True)
    report(args, res, boots, skew)


def report(args, res, boots, skew) -> None:
    gens = GENERATORS

    def mean(d, r, c, p=0.5):
        if p == 0.5:
            return float(np.mean([res[(g, d, r, c)]["balanced_accuracy"] for g in gens]))
        return float(np.mean([skew[(g, d, r, c, p)] for g in gens]))

    lines = ["# Threshold adaptation: how much of the degradation loss is a threshold shift?",
             "Fixed detectors, five threshold rules. **clean val** = the main runs' rule (clean seen-generator "
             "validation); **degraded val** = re-picked on the seen generators' validation images degraded like the "
             "test condition (needs the condition; E4's +recal); **test median** and **test GMM** = label-free, from "
             "the unlabelled test batch (one held-out generator, one condition): its median score, or the crossing "
             "point of a two-component Gaussian mixture on its logit scores; **oracle** = best threshold on the test "
             "labels (upper bound). Mean balanced accuracy over the 7 held-out generators; best non-oracle rule per "
             "detector and column in bold."]
    rows = ["| Detector | Rule | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |",
            "|---|---|" + "---:|" * (len(CONDS) + 1)]
    for d in DETECTORS:
        best = {c: max(mean(d, r, c) for r in RULES[:-1]) for c in CONDS}
        for r in RULES:
            cells = [f"**{mean(d, r, c):.1%}**" if r != "oracle" and np.isclose(mean(d, r, c), best[c])
                     else f"{mean(d, r, c):.1%}" for c in CONDS]
            rows.append(f"| {d} | {r} | " + " | ".join(cells) + f" | {np.mean([mean(d, r, c) for c in STD]):.1%} |")
    lines += ["## Balanced accuracy, 50% real test batches", "\n".join(rows)]

    rows = ["| Detector | Rule − clean val | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |",
            "|---|---|" + "---:|" * (len(CONDS) + 1)]
    for d in DETECTORS:
        for r in RULES[1:]:
            cells = []
            for c in CONDS + ["_mean"]:
                cs = STD if c == "_mean" else [c]
                point = np.mean([res[(g, d, r, x)]["balanced_accuracy"] - res[(g, d, "clean val", x)]["balanced_accuracy"]
                                 for g in gens for x in cs])
                rep = np.mean([boots[(g, d, r, x)]["balanced_accuracy"] - boots[(g, d, "clean val", x)]["balanced_accuracy"]
                               for g in gens for x in cs], axis=0)
                cells.append(cpe.fmt_diff(point, rep, 100, 1))
            rows.append(f"| {d} | {r} | " + " | ".join(cells) + " |")
    lines += [f"## Gain over the clean-val threshold (pp), paired bootstrap ({args.bootstrap} replicates per holdout, "
              "test-batch thresholds held fixed); * = 95% interval excludes 0", "\n".join(rows)]

    rows = ["| Detector | Rule | " + " | ".join(f"{int(p * 100)}% real" for p in PRIORS) + " |", "|---|---|" + "---:|" * len(PRIORS)]
    for d in DETECTORS:
        for r in RULES:
            rows.append(f"| {d} | {r} | " + " | ".join(f"{np.mean([mean(d, r, c, p) for c in STD]):.1%}" for p in PRIORS) + " |")
    lines += [f"## Sensitivity to the share of real images in the test batch: mean BA over the six standard "
              f"conditions ({args.draws} random minority subsets per skewed batch)", "\n".join(rows)]

    rows = ["| Detector | Rule | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|---|" + "---:|" * len(CONDS)]
    for d in (rd.DINO, rd.RN_SQ):
        for r in RULES:
            rows.append(f"| {d} | {r} | " + " | ".join(
                f"{np.mean([res[(g, d, r, c)]['real_recall'] for g in gens]):.0%} / "
                f"{np.mean([res[(g, d, r, c)]['generated_recall'] for g in gens]):.0%}" for c in CONDS) + " |")
    lines += ["## Class recall (real / generated), 50% real batches", "\n".join(rows)]
    cpe.write_report("threshold_adaptation", lines,
                     {f"{g}|{d}|{r}|{c}": v for (g, d, r, c), v in res.items()}
                     | {f"skew|{g}|{d}|{r}|{c}|{p}": v for (g, d, r, c, p), v in skew.items()})


if __name__ == "__main__":
    main()
