"""Follow-up probe experiments that reuse cached frozen-DINOv2 embeddings (no backbone pass).

E3  leave-one-family-out
    SD v1.5 and Wukong are both Stable Diffusion models, so holding out one of them still
    trains on its sibling. For each SD target we compare three training sets:
      A  sibling seen   - the standard leave-one-generator-out run (6 generators)
      B  family out     - both SD generators removed (5 generators)
      C  size control   - sibling kept, one non-SD generator removed instead (5 generators,
                          averaged over the 5 possible choices)
    B - A mixes "sibling removed" with "one generator fewer"; B - C isolates the sibling.
    Real images are always the same BigGAN-folder photos, so only the fake set changes.

E4  degradation-aware routing between two probes
    The clean+diff MLP is best on clean images, the clean-only logistic regression under
    strong degradation. For every held-out generator this compares, per test condition:
      mlp_clean+diff, logreg_clean, fusion  - threshold from the clean seen-generator
                                               validation split (as in the main runs)
      ... +recal                             - same score, threshold re-picked on seen
                                               generators degraded the same way
      router                                 - per condition, the probe (or fusion) and
                                               threshold that are best on seen generators
                                               degraded the same way
    "Degraded seen-generator validation" = the six seen generators' val images (500 real +
    500 generated each) under the test condition; they are cached because every generator
    was a held-out test set once. The router therefore needs the degradation type at test
    time (image size and JPEG quality, both readable from the file), but never sees the
    held-out generator.

A1  leave-one-family-out for every family
    E3 for all generators. Families: GAN (BigGAN), VQ diffusion (VQDM), latent diffusion
    (SD v1.5, Wukong), OpenAI pixel-space diffusion (ADM, GLIDE), Midjourney. For each
    generator the "family out" training set drops its whole family (= the LOGO run for a
    single-generator family), giving a family-level headline number; multi-generator families
    also get E3's A/B/C size-controlled comparison.

A4  MLP training-seed robustness
    The clean+diff MLP of every LOGO holdout refit with several seeds (init and early-stopping
    split only), together with the E4 fusion and router built on it, to check that the 1-5 pp
    differences we report exceed seed noise.

All experiments refuse to embed anything: a missing cache file is an error.

    python scripts/cached_probe_experiments.py e3
    python scripts/cached_probe_experiments.py e4
    python scripts/cached_probe_experiments.py a1
    python scripts/cached_probe_experiments.py a4 --seeds 42 1 2 3 4
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dinov2_genimage_holdout as dh  # noqa: E402
from compare_holdout_results import bootstrap_metrics, ci  # noqa: E402

resnet = dh.resnet
GENERATORS = resnet.GENERATORS
SHORT = {"imagenet_ai_0419_biggan": "BigGAN", "imagenet_ai_0419_vqdm": "VQDM", "imagenet_ai_0424_sdv5": "SD v1.5",
         "imagenet_ai_0424_wukong": "Wukong", "imagenet_ai_0508_adm": "ADM", "imagenet_glide": "GLIDE",
         "imagenet_midjourney": "Midjourney"}
SD_FAMILY = ["imagenet_ai_0424_sdv5", "imagenet_ai_0424_wukong"]
REAL_SOURCE = "imagenet_ai_0419_biggan"
FAMILIES = {"GAN": ["imagenet_ai_0419_biggan"], "VQ diffusion": ["imagenet_ai_0419_vqdm"],
            "Latent diffusion (Stable Diffusion)": SD_FAMILY,
            "Pixel-space diffusion (OpenAI)": ["imagenet_ai_0508_adm", "imagenet_glide"],
            "Midjourney": ["imagenet_midjourney"]}
COND_LABEL = {"clean": "Clean", "jpeg_q50": "JPEG Q50", "resize_128": "Resize 128", "resize_64": "Resize 64",
              "resize_32": "Resize 32", "resize_jpeg_128": "128 + JPEG 50"}


# ------------------------------------------------------------------------ helpers

def best_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Vectorised resnet.best_threshold: the observed score t maximising balanced accuracy
    of (s >= t), first maximum in ascending order, so it picks the same threshold."""
    order = np.argsort(s, kind="stable")
    ys, ss = y[order].astype(bool), s[order]
    uniq, first = np.unique(ss, return_index=True)
    real_below = np.concatenate([[0], np.cumsum(ys)])[first]
    n_real, n_fake = ys.sum(), (~ys).sum()
    ba = 0.5 * ((n_real - real_below) / n_real + (first - real_below) / n_fake)
    i = int(np.argmax(ba))
    return float(uniq[i]), float(ba[i])


def metrics(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pred = s >= t
    real, fake = y == 1, y == 0
    return {"balanced_accuracy": float(0.5 * (pred[real].mean() + (~pred[fake]).mean())),
            "auroc_real_positive": float(roc_auc_score(y, s)),
            "real_recall": float(pred[real].mean()), "generated_recall": float((~pred[fake]).mean()),
            "threshold": t}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def splits(args, seen: list[str], real_source: str):
    """Fit / seen-validation records exactly as run_holdout builds them, with a chosen real-photo folder."""
    real = resnet.files_for(args.data_root, real_source, "train", "nature")
    random.Random(args.seed).shuffle(real)
    k = max(1, len(real) // 5)
    fit = resnet.records_for(args.data_root, seen, "train", real_paths=real[k:])
    val = resnet.records_for(args.data_root, seen, "val", real_paths=real[:k])
    return fit, val


def test_records(args, generator: str) -> list:
    return resnet.records_for(args.test_data_root, [generator], args.test_split)


class Boot:
    """Paired stratified bootstrap indices over one test set (reals first, then fakes)."""

    def __init__(self, y: np.ndarray, n: int, rng: np.random.Generator):
        real, fake = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        self.n_real = len(real)
        self.idx = np.concatenate([rng.choice(real, (n, len(real))), rng.choice(fake, (n, len(fake)))], axis=1)

    def __call__(self, s: np.ndarray, t: float) -> dict:
        return bootstrap_metrics({"score": s, "pred_real": s >= t}, self.idx, self.n_real)


def fmt_diff(point: float, reps: np.ndarray, scale: float, d: int) -> str:
    lo, hi = ci(reps)
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{point * scale:+.{d}f}{star} [{lo * scale:+.{d}f}, {hi * scale:+.{d}f}]"


def make_store(args) -> dh.EmbeddingStore:
    store = dh.EmbeddingStore(args)

    def refuse():
        raise SystemExit("A needed embedding is not cached; these experiments only reuse cached embeddings. "
                         "Run scripts/dinov2_genimage_holdout.py for the missing holdout first.")
    store.model = refuse
    return store


# ------------------------------------------------------------------- E3 and A1

def family_settings(args, store, target: str, family: list[str], rng, results: dict, boots: dict) -> None:
    """Fit every detector on each training set for one target generator and test it.

    A = all generators but the target; for a family with siblings also B = the whole family
    out and C:x = target and one generator x outside the family out (size control). Real
    training photos stay the BigGAN-folder ones (VQDM's when BigGAN is the target), which is
    what the main runs use for setting A."""
    others = [g for g in GENERATORS if g not in family]
    settings = {"A": [g for g in GENERATORS if g != target]}
    if len(family) > 1:
        settings["B"] = others
        for x in others:
            settings[f"C:{x}"] = [g for g in GENERATORS if g not in (target, x)]
    real_source = REAL_SOURCE if target != REAL_SOURCE else settings["A"][0]
    tests = {c: store.gather(test_records(args, target), c, "none") for c in dh.CONDITIONS}
    boot = {c: Boot(tests[c]["label"].astype(int), args.bootstrap, rng) for c in dh.CONDITIONS}
    for setting, seen in settings.items():
        fit_rec, val_rec = splits(args, seen, real_source)
        fit, val = store.gather(fit_rec, "clean", "none"), store.gather(val_rec, "clean", "none")
        for det in dh.DETECTORS:
            t0 = time.time()
            score_fn, _ = dh.fit_detector(det, fit, args, target)
            thr, val_ba = best_threshold(val["label"].astype(int), score_fn(val))
            for c in dh.CONDITIONS:
                s = score_fn(tests[c])
                results[(target, setting, det, c)] = metrics(tests[c]["label"].astype(int), s, thr)
                boots[(target, setting, det, c)] = boot[c](s, thr)
            print(f"{SHORT[target]:8s} {setting:32s} {det:18s} seen-val BA {val_ba:.3f}  "
                  f"clean BA {results[(target, setting, det, 'clean')]['balanced_accuracy']:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)


def abc_tables(args, results, boots, targets: list[str], families: dict) -> list[str]:
    lines = []
    for metric, title, scale, d, f in [("balanced_accuracy", "Balanced accuracy", 100, 1, "{:.1%}"),
                                       ("auroc_real_positive", "AUROC", 1, 3, "{:.3f}")]:
        unit = " (pp)" if scale == 100 else ""
        for cond_set, cond_title in [(["clean"], "clean test images"), (dh.CONDITIONS, "mean over the 6 test conditions")]:
            rows = [f"| Target | Detector | A sibling seen | B family out | C size control | B − A{unit} | B − C{unit} |",
                    "|---|---|---:|---:|---:|---:|---:|"]
            for target in targets:
                C_keys = [f"C:{x}" for x in GENERATORS if x not in families[target]]
                for det in dh.DETECTORS:
                    def point(setting):
                        return np.mean([results[(target, setting, det, c)][metric] for c in cond_set])

                    def reps(setting):
                        return np.mean([boots[(target, setting, det, c)][metric] for c in cond_set], axis=0)
                    a, b = point("A"), point("B")
                    cvals = [point(k) for k in C_keys]
                    c_mean = float(np.mean(cvals))
                    c_reps = np.mean([reps(k) for k in C_keys], axis=0)
                    rows.append(f"| {SHORT[target]} | {det} | {f.format(a)} | {f.format(b)} | "
                                f"{f.format(c_mean)} [{f.format(min(cvals))}–{f.format(max(cvals))}] | "
                                f"{fmt_diff(b - a, reps('B') - reps('A'), scale, d)} | "
                                f"{fmt_diff(b - c_mean, reps('B') - c_reps, scale, d)} |")
            lines += [f"## {title}, {cond_title}", "\n".join(rows)]
    return lines


def run_e3(args, store) -> None:
    results, boots = {}, {}
    rng = np.random.default_rng(args.boot_seed)
    for target in SD_FAMILY:
        family_settings(args, store, target, SD_FAMILY, rng, results, boots)
    lines = ["# E3: leave-one-family-out (Stable Diffusion family)",
             "SD v1.5 and Wukong are both Stable Diffusion models. **A** = standard leave-one-generator-out "
             "(the SD sibling is in training, 6 generators). **B** = both SD generators removed (5 generators). "
             "**C** = sibling kept but one non-SD generator removed (5 generators), averaged over the 5 choices "
             "(range in brackets). B − A mixes 'no sibling' with 'one generator fewer'; B − C isolates the sibling. "
             "All settings use the same real training photos; thresholds come from each setting's own "
             f"seen-generator validation split. Differences: paired stratified bootstrap ({args.bootstrap} "
             "replicates) on the 500 + 500 test images; * = 95% interval excludes 0."]
    lines += abc_tables(args, results, boots, SD_FAMILY, {t: SD_FAMILY for t in SD_FAMILY})
    write_report("e3_family_holdout", lines, {f"{t}|{s}|{d}|{c}": v for (t, s, d, c), v in results.items()})


def run_a1(args, store) -> None:
    results, boots = {}, {}
    rng = np.random.default_rng(args.boot_seed)
    family_of = {g: members for members in FAMILIES.values() for g in members}
    for g in GENERATORS:
        family_settings(args, store, g, family_of[g], rng, results, boots)
    multi = [g for g in GENERATORS if len(family_of[g]) > 1]
    out = {g: "B" if len(family_of[g]) > 1 else "A" for g in GENERATORS}

    lines = ["# A1: leave-one-family-out for every generator family",
             "Families: " + "; ".join(f"**{name}** = {', '.join(SHORT[g] for g in m)}" for name, m in FAMILIES.items())
             + ". **LOGO** = the standard leave-one-generator-out run (6 seen generators). **Family out** = the "
             "target's whole family removed from training; for single-generator families it is the LOGO run itself. "
             "Real training photos are the same in every setting; thresholds come from each setting's own "
             "seen-generator validation split. Differences: paired stratified bootstrap "
             f"({args.bootstrap} replicates per held-out generator), averaged over the 7 generators; "
             "* = 95% interval excludes 0."]
    for metric, title, scale, d, f in [("balanced_accuracy", "Balanced accuracy", 100, 1, "{:.1%}"),
                                       ("auroc_real_positive", "AUROC", 1, 3, "{:.3f}")]:
        rows = ["| Detector | Training set | " + " | ".join(COND_LABEL[c] for c in dh.CONDITIONS) + " | Mean |",
                "|---|---|" + "---:|" * (len(dh.CONDITIONS) + 1)]
        for det in dh.DETECTORS:
            for name, pick in [("LOGO", lambda g: "A"), ("family out", lambda g: out[g])]:
                vals = {c: [results[(g, pick(g), det, c)][metric] for g in GENERATORS] for c in dh.CONDITIONS}
                cells = [f.format(np.mean(v)) + f" [{f.format(min(v))}]".replace("%]", "]") for v in vals.values()]
                rows.append(f"| {det} | {name} | " + " | ".join(cells)
                            + f" | {f.format(np.mean(list(vals.values())))} |")
            diffs = []
            for c in dh.CONDITIONS:
                point = np.mean([results[(g, out[g], det, c)][metric] - results[(g, 'A', det, c)][metric] for g in GENERATORS])
                rep = np.mean([boots[(g, out[g], det, c)][metric] - boots[(g, 'A', det, c)][metric] for g in GENERATORS], axis=0)
                diffs.append(fmt_diff(point, rep, scale, d))
            rows.append(f"| {det} | family out − LOGO | " + " | ".join(diffs) + " | |")
        lines += [f"## {title}: 7-generator mean [worst generator]", "\n".join(rows)]

    rows = ["| Held-out generator | Family | " + " | ".join(f"{det} LOGO → family out" for det in dh.DETECTORS) + " |",
            "|---|---|" + "---:|" * len(dh.DETECTORS)]
    fam_name = {g: name for name, m in FAMILIES.items() for g in m}
    for g in GENERATORS:
        rows.append(f"| {SHORT[g]} | {fam_name[g]} | " + " | ".join(
            f"{results[(g, 'A', det, 'clean')]['balanced_accuracy']:.1%} → {results[(g, out[g], det, 'clean')]['balanced_accuracy']:.1%}"
            for det in dh.DETECTORS) + " |")
    lines += ["## Clean balanced accuracy per held-out generator", "\n".join(rows)]
    lines += ["## Families with more than one generator: sibling vs. training-set size",
              "**A** = LOGO (sibling seen), **B** = family out, **C** = sibling kept but one generator from another "
              "family removed instead, averaged over the possible choices (range in brackets); B − C isolates the "
              "sibling from having one generator fewer."]
    lines += abc_tables(args, results, boots, multi, family_of)
    write_report("a1_family_holdout", lines, {f"{t}|{s}|{d}|{c}": v for (t, s, d, c), v in results.items()})


# --------------------------------------------------------------------- E4 and A4

ROUTE_FROM = ("mlp_clean+diff", "logreg_clean", "fusion")


def holdout_data(args, store, held_out: str):
    """Fit, clean seen-val, degraded seen-val (seen generators' test folders) and test embeddings."""
    seen = [g for g in GENERATORS if g != held_out]
    fit_rec, val_rec = splits(args, seen, seen[0])          # identical to the main LOGO runs
    fit, val = store.gather(fit_rec, "clean", "none"), store.gather(val_rec, "clean", "none")
    dval = {c: store.gather([r for g in seen for r in test_records(args, g)], c, "none") for c in dh.CONDITIONS}
    tests = {c: store.gather(test_records(args, held_out), c, "none") for c in dh.CONDITIONS}
    overlap = {Path(p).name for p in dval["clean"]["path"]} & {Path(p).name for p in tests["clean"]["path"]}
    assert not overlap, f"degraded validation shares {len(overlap)} files with the {held_out} test set"
    return fit, val, dval, tests


def route(probes: dict, val, dval, tests) -> tuple[dict, dict]:
    """Per condition: {method: (test scores, threshold)} and the router's choice.

    Plain methods use the clean seen-val threshold, '+recal' the degraded seen-val one; the
    router takes whichever of ROUTE_FROM is best on the degraded seen-val images."""
    def scores(emb):
        s = {d: fn(emb) for d, fn in probes.items()}
        s["fusion"] = 0.5 * (logit(s["mlp_clean+diff"]) + logit(s["logreg_clean"]))
        return s
    yv = val["label"].astype(int)
    clean_thr = {k: best_threshold(yv, v)[0] for k, v in scores(val).items()}
    plans, picks = {}, {}
    for c in dh.CONDITIONS:
        sd, st = scores(dval[c]), scores(tests[c])
        recal = {k: best_threshold(dval[c]["label"].astype(int), v) for k, v in sd.items()}
        choice = max(ROUTE_FROM, key=lambda k: recal[k][1])
        picks[c] = choice
        plans[c] = {**{k: (st[k], clean_thr[k]) for k in st},
                    **{f"{k} +recal": (st[k], recal[k][0]) for k in st},
                    "router": (st[choice], recal[choice][0])}
    return plans, picks


def run_e4(args, store) -> None:
    rng = np.random.default_rng(args.boot_seed)
    methods = ["mlp_clean+diff", "logreg_clean", "fusion", "mlp_clean+diff +recal", "logreg_clean +recal",
               "fusion +recal", "router"]
    results, boots, picks = {}, {}, {}
    for held_out in GENERATORS:
        fit, val, dval, tests = holdout_data(args, store, held_out)
        t0 = time.time()
        probes = {d: dh.fit_detector(d, fit, args, held_out)[0] for d in ("mlp_clean+diff", "logreg_clean")}
        plans, chosen = route(probes, val, dval, tests)
        for c in dh.CONDITIONS:
            yt = tests[c]["label"].astype(int)
            picks[(held_out, c)] = chosen[c]
            boot = Boot(yt, args.bootstrap, rng)
            for m in methods:
                s, t = plans[c][m]
                results[(held_out, m, c)] = metrics(yt, s, t)
                boots[(held_out, m, c)] = boot(s, t)
        print(f"E4 {SHORT[held_out]:10s} clean BA mlp {results[(held_out, 'mlp_clean+diff', 'clean')]['balanced_accuracy']:.3f} "
              f"logreg {results[(held_out, 'logreg_clean', 'clean')]['balanced_accuracy']:.3f}  "
              f"router picks {[picks[(held_out, c)] for c in dh.CONDITIONS]}  ({time.time() - t0:.0f}s)")

    # ---- report ----
    gens = GENERATORS
    lines = ["# E4: degradation-aware routing between the clean-only and clean+diff probes",
             "Mean over the 7 held-out generators (worst generator in brackets). Plain rows use the threshold "
             "from the clean seen-generator validation split, as in the main runs. **+recal** re-picks only the "
             "threshold on the six seen generators' validation images degraded like the test condition. "
             "**router** also picks, per condition, which score to use (mlp_clean+diff, logreg_clean or their "
             "logit-average **fusion**) by balanced accuracy on those degraded seen-generator images; it needs "
             "the degradation type at test time but never sees the held-out generator."]
    for metric, title, f in [("balanced_accuracy", "Balanced accuracy", "{:.1%}"), ("auroc_real_positive", "AUROC", "{:.3f}")]:
        rows = ["| Method | " + " | ".join(COND_LABEL[c] for c in dh.CONDITIONS) + " | Mean |",
                "|---|" + "---:|" * (len(dh.CONDITIONS) + 1)]
        shown = methods if metric == "balanced_accuracy" else ["mlp_clean+diff", "logreg_clean", "fusion"]
        best = {c: max(np.mean([results[(g, m, c)][metric] for g in gens]) for m in shown) for c in dh.CONDITIONS}
        for m in shown:
            cells = []
            for c in dh.CONDITIONS:
                vals = [results[(g, m, c)][metric] for g in gens]
                text = f.format(np.mean(vals))
                cells.append((f"**{text}**" if np.isclose(np.mean(vals), best[c]) else text)
                             + f" [{f.format(min(vals))}]".replace("%]", "]"))
            overall = np.mean([results[(g, m, c)][metric] for g in gens for c in dh.CONDITIONS])
            rows.append(f"| {m} | " + " | ".join(cells) + f" | {f.format(overall)} |")
        lines += [f"## {title}", "\n".join(rows)]

    for ref in ["mlp_clean+diff", "logreg_clean"]:
        rows = [f"| Method − {ref} | " + " | ".join(COND_LABEL[c] for c in dh.CONDITIONS) + " |",
                "|---|" + "---:|" * len(dh.CONDITIONS)]
        for m in ["fusion", "router", f"{ref} +recal"]:
            cells = []
            for c in dh.CONDITIONS:
                point = np.mean([results[(g, m, c)]["balanced_accuracy"] - results[(g, ref, c)]["balanced_accuracy"] for g in gens])
                reps = np.mean([boots[(g, m, c)]["balanced_accuracy"] - boots[(g, ref, c)]["balanced_accuracy"] for g in gens], axis=0)
                cells.append(fmt_diff(point, reps, 100, 1))
            rows.append(f"| {m} | " + " | ".join(cells) + " |")
        lines += [f"## Mean balanced-accuracy difference vs `{ref}` (pp), 7-generator average, "
                  f"paired bootstrap ({args.bootstrap} replicates per holdout)", "\n".join(rows)]

    names = {"mlp_clean+diff": "MLP", "logreg_clean": "LR", "fusion": "fusion"}
    rows = ["| Held-out generator | " + " | ".join(COND_LABEL[c] for c in dh.CONDITIONS) + " |",
            "|---|" + "---|" * len(dh.CONDITIONS)]
    rows += [f"| {SHORT[g]} | " + " | ".join(names[picks[(g, c)]] for c in dh.CONDITIONS) + " |" for g in gens]
    lines += ["## Score chosen by the router (on degraded seen-generator validation)", "\n".join(rows)]
    write_report("e4_probe_routing", lines, {f"{g}|{m}|{c}": v for (g, m, c), v in results.items()}
                 | {f"pick|{g}|{c}": v for (g, c), v in picks.items()})



def run_a4(args, store) -> None:
    """MLP training-seed robustness of the main LOGO results and of the E4 router."""
    seeds = args.seeds
    fixed = ["logreg_clean", "logreg_clean+diff"]
    varying = ["mlp_clean+diff", "fusion", "router"]
    res = {}                                           # (seed, generator, method, condition) -> metrics
    for g in GENERATORS:
        fit, val, dval, tests = holdout_data(args, store, g)
        lr = {d: dh.fit_detector(d, fit, args, g)[0] for d in fixed}
        for seed in seeds:
            t0 = time.time()
            a = copy.copy(args)
            a.seed = seed                              # only the MLP's init / early-stopping split change
            probes = {"mlp_clean+diff": dh.fit_detector("mlp_clean+diff", fit, a, g)[0], **lr}
            plans, _ = route(probes, val, dval, tests)
            for c in dh.CONDITIONS:
                yt = tests[c]["label"].astype(int)
                for m in fixed + varying:
                    res[(seed, g, m, c)] = metrics(yt, *plans[c][m])
            print(f"A4 {SHORT[g]:10s} seed {seed:3d} mlp clean BA "
                  f"{res[(seed, g, 'mlp_clean+diff', 'clean')]['balanced_accuracy']:.3f}  ({time.time() - t0:.0f}s)",
                  flush=True)
        if args.seed in seeds:                         # reproduction check against the main run
            run = ROOT / "results" / f"dinov2_mlp_clean_diff_holdout_{g}" / "summary.json"
            saved = json.loads(run.read_text())["test_metrics"]["clean"]["balanced_accuracy"]
            got = res[(args.seed, g, "mlp_clean+diff", "clean")]["balanced_accuracy"]
            print(("  reproduces" if abs(saved - got) < 1e-9 else "  WARNING: differs from") + f" the saved run ({saved:.4f})")

    def mean7(seed, m, c, metric="balanced_accuracy"):
        return np.mean([res[(seed, g, m, c)][metric] for g in GENERATORS])

    def spread(vals, f, scale=1):
        v = np.asarray(vals) * scale
        return f"{f.format(v.mean())} ± {f.format(v.std(ddof=1))} [{f.format(v.min())}, {f.format(v.max())}]"

    lines = ["# A4: MLP training-seed robustness",
             f"The clean+diff MLP was refit with {len(seeds)} training seeds ({', '.join(map(str, seeds))}; 42 is the "
             "main runs' seed). The seed sets the network initialisation and the 10% early-stopping split; data split, "
             "embeddings and the two logistic regressions (deterministic) are unchanged. **fusion** and **router** are "
             "the E4 methods rebuilt on each seed's MLP. Cells: mean ± standard deviation over seeds [min, max] of the "
             "7-generator mean."]
    for metric, title, f, scale in [("balanced_accuracy", "Balanced accuracy (%)", "{:.1f}", 100),
                                    ("auroc_real_positive", "AUROC", "{:.3f}", 1)]:
        rows = ["| Method | " + " | ".join(COND_LABEL[c] for c in dh.CONDITIONS) + " |", "|---|" + "---:|" * len(dh.CONDITIONS)]
        for m in fixed + varying:
            if metric != "balanced_accuracy" and m == "router":
                continue
            if m in fixed:
                rows.append(f"| {m} (deterministic) | " + " | ".join(f.format(mean7(seeds[0], m, c, metric) * scale)
                                                                    for c in dh.CONDITIONS) + " |")
            else:
                rows.append(f"| {m} | " + " | ".join(spread([mean7(s, m, c, metric) for s in seeds], f, scale)
                                                     for c in dh.CONDITIONS) + " |")
        lines += [f"## {title}, 7-generator mean", "\n".join(rows)]

    claims = [("mlp_clean+diff", "logreg_clean"), ("mlp_clean+diff", "logreg_clean+diff"),
              ("router", "mlp_clean+diff"), ("router", "logreg_clean"), ("fusion", "mlp_clean+diff")]
    rows = ["| Comparison | " + " | ".join(COND_LABEL[c] for c in dh.CONDITIONS) + " |", "|---|" + "---:|" * len(dh.CONDITIONS)]
    for a, b in claims:
        cells = []
        for c in dh.CONDITIONS:
            d = [100 * (mean7(s, a, c) - mean7(s, b, c)) for s in seeds]
            same = sum(np.sign(x) == np.sign(np.mean(d)) for x in d)
            cells.append(f"{np.mean(d):+.1f} ± {np.std(d, ddof=1):.1f} ({same}/{len(seeds)})")
        rows.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += ["## Key differences in 7-generator mean balanced accuracy (pp): mean ± SD over seeds "
              "(seeds agreeing in sign with the mean)", "\n".join(rows)]

    rows = ["| Held-out generator | Clean BA (%) | Clean AUROC | Resize 32 BA (%) |", "|---|---:|---:|---:|"]
    for g in GENERATORS:
        rows.append(f"| {SHORT[g]} | {spread([res[(s, g, 'mlp_clean+diff', 'clean')]['balanced_accuracy'] for s in seeds], '{:.1f}', 100)} | "
                    f"{spread([res[(s, g, 'mlp_clean+diff', 'clean')]['auroc_real_positive'] for s in seeds], '{:.3f}')} | "
                    f"{spread([res[(s, g, 'mlp_clean+diff', 'resize_32')]['balanced_accuracy'] for s in seeds], '{:.1f}', 100)} |")
    lines += ["## mlp_clean+diff per held-out generator across seeds", "\n".join(rows)]
    write_report("a4_seed_robustness", lines, {f"{s}|{g}|{m}|{c}": v for (s, g, m, c), v in res.items()})


# --------------------------------------------------------------------------- main

def write_report(name: str, lines: list[str], data: dict) -> None:
    text = "\n\n".join(lines) + "\n"
    (ROOT / "results" / f"{name}.md").write_text(text, encoding="utf-8")
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("\n" + text + f"\nSaved -> results/{name}.md and .json")


def set_defaults(args) -> None:
    """The settings the cached embeddings and the main runs were made with (kept if already set)."""
    defaults = dict(data_root=ROOT / resnet.DEFAULT_DATA_ROOT, test_data_root=ROOT / resnet.DEFAULT_TEST_DATA_ROOT,
                    test_split="test", model_name="dinov2_vitl14", preprocess="crop", lamb=0.05, seed=42,
                    max_per_group=None, C=0.1, hidden=256, device=None, batch_size=32, workers=0)
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v.resolve() if isinstance(v, Path) else v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["e3", "e4", "a1", "a4"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4], help="a4: MLP training seeds")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    set_defaults(args)
    store = make_store(args)
    {"e3": run_e3, "e4": run_e4, "a1": run_a1, "a4": run_a4}[args.experiment](args, store)


if __name__ == "__main__":
    main()
