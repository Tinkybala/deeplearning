"""Final detector: block-20 probes on augmented images, with batch and single-image thresholds.

Combines the three improvements found so far and removes the batch requirement of the label-
free threshold:

  features   DINOv2 block 20 (backbone_comparison; confirmed by nested_selection)
  training   clean, augmented ("both": JPEG 30-95 and/or resize to 32/64/128, p = 0.5 each)
             or clean + augmented (augmented_probe_experiments); the block-20 features of
             the augmented images come from augmented_backbone_features.py
  threshold  clean val      clean seen-generator validation (main runs)
             degraded val   re-picked on seen-generator validation degraded like the test
                            condition (needs the condition)
             metadata A/B   single-image: the threshold of the image's metadata bucket,
                            learned on the degraded seen-generator validation images in the
                            same bucket. Bucket = shorter side of the received file (< 48,
                            < 96, else full size) and, for A, whether it is a JPEG of
                            quality <= 75 (luminance quantisation table). B uses size only.
             test GMM       label-free, from a batch of unlabelled test images
                            (threshold_adaptation_experiments.py)
             oracle         best threshold on the test labels (upper bound)
  routers    condition router (E4 / backbone_comparison, needs the condition) and metadata
             routers A/B (per bucket, the probe and threshold best on that bucket's degraded
             validation images), each over every probe here and in backbone_comparison

The received file is modelled as the last encoding of each simulated condition: the original
file for clean images (real photos are JPEGs of mixed quality, generated images PNG), JPEG
Q50 for the JPEG conditions, JPEG Q96 for the format-matched one, and a lossless file of the
reduced size after resizing. In GenImage, file metadata also carries class information
(generated images are never JPEG; BigGAN's are all 128 px), so the buckets are coarse; the
report lists each bucket's real / generated composition per condition.

    python scripts/final_detector_experiments.py
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_comparison as bc  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402
import threshold_adaptation_experiments as ta  # noqa: E402

dh = bc.dh
GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
CONDS, STD, LABEL = bc.CONDS, dh.CONDITIONS, bc.LABEL
AUG = "clean__aug-both"
L20 = "DINOv2 L20 MLP"
NEW = {f"{L20} [both]": ("MLP", "both"), f"{L20} [clean+both]": ("MLP", "clean+both"),
       "DINOv2 L20 LR [both]": ("LR", "both"), "DINOv2 L20 LR [clean+both]": ("LR", "clean+both")}
REF_AUG = f"{bc.REF} [clean+both]"
DETECTORS = [bc.REF, REF_AUG, L20, f"{L20} [both]", f"{L20} [clean+both]", bc.FUSION]
RULES = ["clean val", "degraded val", "metadata A", "metadata B", "test GMM", "oracle"]
ROUTERS = ["condition router", "metadata router A", "metadata router B"]
LOWQ = 29.03                           # mean luminance quantisation step of a PIL JPEG at quality 75


@lru_cache(maxsize=None)
def file_info(path: str) -> tuple[int, bool]:
    """(shorter side, is a JPEG of quality <= 75) of an original file."""
    with Image.open(path) as im:
        lowq = im.format == "JPEG" and float(np.mean(im.quantization[0])) >= LOWQ
        return min(im.size), lowq


def received(path: str, condition: str) -> tuple[int, bool]:
    side, lowq = file_info(path)
    if condition in ("jpeg_q50", "resize_jpeg_128"):
        lowq = True
    elif condition.startswith("fmt_") or condition.startswith("resize_"):
        lowq = False
    if condition.startswith("resize_"):
        side = int(condition.rsplit("_", 1)[1])
    elif condition == "fmt_r256_q96":
        side = 256
    return side, lowq


def bucket(path: str, condition: str, use_quality: bool) -> str:
    side, lowq = received(path, condition)
    size = "<48" if side < 48 else "<96" if side < 96 else "full"
    return size + ("|jpeg<=75" if use_quality and lowq else "")


def bucket_threshold(y, s, b, fallback):
    """Per-bucket BA-optimal threshold; buckets with < 20 images of a class use the fallback."""
    thr = {}
    for k in np.unique(b):
        m = b == k
        thr[k] = cpe.best_threshold(y[m], s[m])[0] if min((y[m] == 1).sum(), (y[m] == 0).sum()) >= 20 else fallback
    return thr


def scores_for(args, fstore, dstore, g: str) -> tuple[dict, dict, dict]:
    seen = [x for x in GENERATORS if x != g]
    S = bc.scores_for(args, fstore, dstore, g, seen, with_degraded=True)
    y = S.pop("_y")
    fit_rec, val_rec = cpe.splits(args, seen, seen[0])
    val_new = bc.fa.shared_val(args, val_rec)
    test_rec = cpe.test_records(args, g)
    dval_rec = [r for x in seen for r in cpe.test_records(args, x)]
    yf = np.asarray([r[1] for r in fit_rec])
    keys = ["dino_cls20"]
    Xc, Xa = fstore.gather(fit_rec, "clean", keys), fstore.gather(fit_rec, AUG, keys)
    Xv = fstore.gather(val_new, "clean", keys)
    tests = {c: fstore.gather(test_rec, c, keys) for c in CONDS}
    dvals = {c: fstore.gather(dval_rec, c, keys) for c in CONDS}
    for name, (kind, train) in NEW.items():
        X, yy = (Xa, yf) if train == "both" else (np.vstack([Xc, Xa]), np.concatenate([yf, yf]))
        fn = bc.fit_probe(kind, X, yy, args)
        S[name] = {"val": fn(Xv), "test": {c: fn(t) for c, t in tests.items()}, "dval": {c: fn(d) for c, d in dvals.items()}}
    # main runs' MLP refit on clean + augmented final-layer embeddings (RIGID cache)
    fit = dstore.gather(fit_rec, "clean", "none")
    aug = dstore.gather(fit_rec, "clean", "both")
    both = {k: np.concatenate([fit[k], aug[k]]) for k in ("clean", "noisy", "label", "group", "path")}
    fn = dh.fit_detector("mlp_clean+diff", both, args, g)[0]
    S[REF_AUG] = {"val": fn(dstore.gather(val_rec, "clean", "none")),
                  "test": {c: fn(dstore.gather(test_rec, c, "none")) for c in CONDS},
                  "dval": {c: fn(dstore.gather(dval_rec, c, "none")) for c in CONDS}}
    paths = {"test": [str(r[0]) for r in test_rec], "dval": [str(r[0]) for r in dval_rec]}
    return S, y, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    fstore, dstore = bc.FeatureStore(), cpe.make_store(args)
    rng, sub_rng = np.random.default_rng(args.boot_seed), np.random.default_rng(1)
    res, boots, skew, picks, comp = {}, {}, {}, {}, {}

    for g in GENERATORS:
        t0 = time.time()
        S, y, paths = scores_for(args, fstore, dstore, g)
        yt, yd = y["test"], y["dval"]
        boot = {c: cpe.Boot(yt, args.bootstrap, rng) for c in CONDS}
        bt = {(q, c): np.asarray([bucket(p, c, q) for p in paths["test"]]) for q in (True, False) for c in CONDS}
        bd = {q: np.concatenate([[bucket(p, c, q) for p in paths["dval"]] for c in CONDS]) for q in (True, False)}
        yd_all = np.concatenate([yd] * len(CONDS))
        for c in CONDS:
            for b in np.unique(bt[(True, c)]):
                m = bt[(True, c)] == b
                comp[(g, c, b)] = (int((yt[m] == 1).sum()), int((yt[m] == 0).sum()))

        def logits(m):
            p = S[m]["val"].min() >= 0 and S[m]["val"].max() <= 1
            return {"val": ta.to_logit(S[m]["val"], p), "test": {c: ta.to_logit(S[m]["test"][c], p) for c in CONDS},
                    "dval": {c: ta.to_logit(S[m]["dval"][c], p) for c in CONDS}}
        L = {m: logits(m) for m in S}
        meta_thr = {(m, q): bucket_threshold(yd_all, np.concatenate([L[m]["dval"][c] for c in CONDS]), bd[q],
                                             cpe.best_threshold(y["val"], L[m]["val"])[0])
                    for m in S for q in (True, False)}

        def record(name, c, margin):                     # margin > 0 means "real"
            res[(g, name, c)] = cpe.metrics(yt, margin, 0.0)
            boots[(g, name, c)] = boot[c](margin, 0.0)

        for d in DETECTORS:
            s = L[d]
            for c in CONDS:
                t = s["test"][c]
                for rule in RULES:
                    if rule.startswith("metadata"):
                        q = rule.endswith("A")
                        thr = np.asarray([meta_thr[(d, q)][b] for b in bt[(q, c)]])
                    else:
                        thr = ta.thresholds(rule, s, c, t, yt, y)
                    record(f"{d} | {rule}", c, t - thr)
                if d.startswith(L20):
                    real, fake = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
                    for p in (0.2, 0.8):
                        acc = []
                        for _ in range(args.draws):
                            k = round(len(fake) * p / (1 - p)) if p < 0.5 else round(len(real) * (1 - p) / p)
                            idx = (np.concatenate([sub_rng.choice(real, k, replace=False), fake]) if p < 0.5
                                   else np.concatenate([real, sub_rng.choice(fake, k, replace=False)]))
                            acc.append(cpe.metrics(yt[idx], t[idx], ta.gmm_threshold(t[idx]))["balanced_accuracy"])
                        skew[(g, d, c, p)] = float(np.mean(acc))

        methods = list(S)
        for c in CONDS:                                   # condition router
            recal = {m: cpe.best_threshold(yd, L[m]["dval"][c]) for m in methods}
            best = max(methods, key=lambda m: recal[m][1])
            picks[(g, "condition router", c)] = best
            record("condition router", c, L[best]["test"][c] - recal[best][0])
        for q, name in ((True, "metadata router A"), (False, "metadata router B")):
            choice = {}
            for b in np.unique(bd[q]):                     # per bucket: best probe on that bucket's dval
                m_ = bd[q] == b
                ok = min((yd_all[m_] == 1).sum(), (yd_all[m_] == 0).sum()) >= 20
                choice[b] = max(methods, key=lambda m: cpe.best_threshold(
                    yd_all[m_], np.concatenate([L[m]["dval"][c] for c in CONDS])[m_])[1]) if ok else L20
            for c in CONDS:
                margin = np.empty(len(yt))
                for b in np.unique(bt[(q, c)]):
                    m_ = bt[(q, c)] == b
                    m = choice.get(b, L20)
                    margin[m_] = L[m]["test"][c][m_] - meta_thr[(m, q)].get(b, cpe.best_threshold(y["val"], L[m]["val"])[0])
                record(name, c, margin)
                picks[(g, name, c)] = "; ".join(f"{b}: {choice.get(b, L20)}" for b in np.unique(bt[(q, c)]))
        print(f"{SHORT[g]:10s} " + "  ".join(
            f"{k} {res[(g, k, 'jpeg_q50')]['balanced_accuracy']:.3f}" for k in
            [f"{L20} | clean val", f"{L20} [clean+both] | clean val", f"{L20} [clean+both] | metadata A",
             f"{L20} [clean+both] | test GMM", "metadata router A"]) + f"  (JPEG50 BA, {time.time() - t0:.0f}s)", flush=True)
    import pickle
    (ROOT / "results" / "_final_detector_state.pkl").write_bytes(pickle.dumps((res, boots, skew, picks, comp)))
    report(args, res, boots, skew, picks, comp)


def rebuild() -> None:
    """Rewrite the report from the saved state, without refitting (python ... --rebuild)."""
    import pickle
    args = argparse.Namespace(bootstrap=2000)
    report(args, *pickle.loads((ROOT / "results" / "_final_detector_state.pkl").read_bytes()))


def report(args, res, boots, skew, picks, comp) -> None:
    gens = GENERATORS

    def mean(m, c):
        return float(np.mean([res[(g, m, c)]["balanced_accuracy"] for g in gens]))

    def row(m, label=None):
        return f"| {label or m} | " + " | ".join(f"{mean(m, c):.1%}" for c in CONDS) + f" | {np.mean([mean(m, c) for c in STD]):.1%} |"

    head = "| Detector | " + " | ".join(LABEL[c] for c in CONDS) + " | Mean (6 std.) |\n|---|" + "---:|" * (len(CONDS) + 1)
    lines = ["# Final detector: block-20 probes, augmented training, batch and single-image thresholds",
             "Mean balanced accuracy over the 7 held-out generators. Training: **[both]** = fit on the augmented "
             "images only (JPEG 30–95 and/or resize to 32/64/128 px, p = 0.5 each), **[clean+both]** = clean and "
             "augmented together, no tag = clean only. Thresholds: **clean val** (main runs), **degraded val** "
             "(needs the test condition), **metadata A** (single image: threshold of its bucket = shorter side < 48 / "
             "< 96 / full and JPEG quality <= 75, learned on degraded seen-generator validation), **metadata B** (size "
             "buckets only), **test GMM** (label-free, needs a batch of test images of one condition), **oracle** (test "
             "labels; upper bound). Routers choose among every probe here and in backbone_comparison, per condition "
             "(needs the condition) or per metadata bucket."]
    for d in DETECTORS:
        lines += [f"## {d}", head + "\n" + "\n".join(row(f"{d} | {r}", r) for r in RULES)]
    lines += ["## Routers", head + "\n" + "\n".join(row(m) for m in ROUTERS)]

    best = f"{L20} | test GMM"
    pairs = [(f"{L20} [clean+both] | clean val", f"{L20} | clean val"),
             (f"{L20} [clean+both] | test GMM", best),
             (f"{L20} [clean+both] | metadata A", best), (f"{L20} [clean+both] | metadata B", best),
             (f"{L20} [clean+both] | metadata A", "condition router"),
             ("metadata router A", "condition router"), ("metadata router B", "condition router"),
             (f"{L20} [clean+both] | test GMM", "condition router")]
    rows = [head.replace("Detector", "Comparison")]
    for a, b in pairs:
        cells = []
        for c in CONDS + ["_mean"]:
            cs = STD if c == "_mean" else [c]
            point = np.mean([res[(g, a, x)]["balanced_accuracy"] - res[(g, b, x)]["balanced_accuracy"] for g in gens for x in cs])
            rep = np.mean([boots[(g, a, x)]["balanced_accuracy"] - boots[(g, b, x)]["balanced_accuracy"]
                           for g in gens for x in cs], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {a.replace(' | ', ', ')} − {b.replace(' | ', ', ')} | " + " | ".join(cells) + " |")
    lines += [f"## Differences in balanced accuracy (pp), paired bootstrap ({args.bootstrap} replicates per holdout; "
              "batch thresholds held fixed); * = 95% interval excludes 0", "\n".join(rows)]

    rows = ["| Detector | 50% real | 20% real | 80% real |", "|---|---:|---:|---:|"]
    for d in [x for x in DETECTORS if x.startswith(L20)]:
        r50 = np.mean([mean(f"{d} | test GMM", c) for c in STD])
        r20, r80 = (np.mean([np.mean([skew[(g, d, c, p)] for g in gens]) for c in STD]) for p in (0.2, 0.8))
        rows.append(f"| {d} | {r50:.1%} | {r20:.1%} | {r80:.1%} |")
    lines += ["## Test GMM on skewed batches: mean BA over the six standard conditions", "\n".join(rows)]

    buckets = sorted({k[2] for k in comp})
    rows = ["| Condition | " + " | ".join(b.replace("|", ", ") for b in buckets) + " |", "|---|" + "---:|" * len(buckets)]
    for c in CONDS:
        cells = []
        for b in buckets:
            r = sum(comp.get((g, c, b), (0, 0))[0] for g in gens)
            f = sum(comp.get((g, c, b), (0, 0))[1] for g in gens)
            cells.append(f"{r} / {f}" if r or f else "")
        rows.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    lines += ["## Metadata A buckets: test images per bucket, real / generated (all 7 holdouts)", "\n".join(rows)]

    rows = ["| Held-out generator | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    rows += [f"| {SHORT[g]} | " + " | ".join(picks[(g, "metadata router A", c)].replace("|", ", ") for c in CONDS) + " |"
             for g in gens]
    lines += ["## Metadata router A choices (bucket: probe)", "\n".join(rows)]
    cpe.write_report("final_detector", lines,
                     {f"{g}|{m}|{c}": v for (g, m, c), v in res.items()}
                     | {f"skew|{g}|{d}|{c}|{p}": v for (g, d, c, p), v in skew.items()}
                     | {f"pick|{g}|{r}|{c}": v for (g, r, c), v in picks.items()}
                     | {f"bucket|{g}|{c}|{b}": v for (g, c, b), v in comp.items()})


if __name__ == "__main__":
    rebuild() if "--rebuild" in sys.argv else main()
