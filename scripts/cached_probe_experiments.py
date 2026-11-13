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

Both experiments refuse to embed anything: a missing cache file is an error.

    python scripts/cached_probe_experiments.py e3
    python scripts/cached_probe_experiments.py e4
"""

from __future__ import annotations

import argparse
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


# ----------------------------------------------------------------------------- E3

def run_e3(args, store) -> None:
    detectors = dh.DETECTORS
    non_sd = [g for g in GENERATORS if g not in SD_FAMILY]
    results, boots = {}, {}
    rng = np.random.default_rng(args.boot_seed)
    for target in SD_FAMILY:
        settings = {"A": [g for g in GENERATORS if g != target], "B": non_sd}
        for x in non_sd:
            settings[f"C:{x}"] = [g for g in GENERATORS if g not in (target, x)]
        tests = {c: store.gather(test_records(args, target), c, "none") for c in dh.CONDITIONS}
        y = tests["clean"]["label"].astype(int)
        boot = {c: Boot(tests[c]["label"].astype(int), args.bootstrap, rng) for c in dh.CONDITIONS}
        for setting, seen in settings.items():
            fit_rec, val_rec = splits(args, seen, REAL_SOURCE)
            fit, val = store.gather(fit_rec, "clean", "none"), store.gather(val_rec, "clean", "none")
            for det in detectors:
                t0 = time.time()
                score_fn, _ = dh.fit_detector(det, fit, args, target)
                thr, val_ba = best_threshold(val["label"].astype(int), score_fn(val))
                for c in dh.CONDITIONS:
                    s = score_fn(tests[c])
                    results[(target, setting, det, c)] = metrics(tests[c]["label"].astype(int), s, thr)
                    boots[(target, setting, det, c)] = boot[c](s, thr)
                print(f"E3 {SHORT[target]:8s} {setting:32s} {det:18s} seen-val BA {val_ba:.3f}  "
                      f"clean BA {results[(target, setting, det, 'clean')]['balanced_accuracy']:.3f}  "
                      f"({time.time() - t0:.0f}s)")
        assert len(y) == 1000

    # ---- report ----
    C_keys = [f"C:{x}" for x in non_sd]
    lines = ["# E3: leave-one-family-out (Stable Diffusion family)",
             "SD v1.5 and Wukong are both Stable Diffusion models. **A** = standard leave-one-generator-out "
             "(the SD sibling is in training, 6 generators). **B** = both SD generators removed (5 generators). "
             "**C** = sibling kept but one non-SD generator removed (5 generators), averaged over the 5 choices "
             "(range in brackets). B − A mixes 'no sibling' with 'one generator fewer'; B − C isolates the sibling. "
             "All settings use the same real training photos; thresholds come from each setting's own "
             f"seen-generator validation split. Differences: paired stratified bootstrap ({args.bootstrap} "
             "replicates) on the 500 + 500 test images; * = 95% interval excludes 0."]
    for metric, title, scale, d, f in [("balanced_accuracy", "Balanced accuracy", 100, 1, "{:.1%}"),
                                       ("auroc_real_positive", "AUROC", 1, 3, "{:.3f}")]:
        unit = " (pp)" if scale == 100 else ""
        for cond_set, cond_title in [(["clean"], "clean test images"), (dh.CONDITIONS, "mean over the 6 test conditions")]:
            rows = [f"| Target | Detector | A sibling seen | B family out | C size control | B − A{unit} | B − C{unit} |",
                    "|---|---|---:|---:|---:|---:|---:|"]
            for target in SD_FAMILY:
                for det in detectors:
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
    write_report("e3_family_holdout", lines, {f"{t}|{s}|{d}|{c}": v for (t, s, d, c), v in results.items()})


# ----------------------------------------------------------------------------- E4

def run_e4(args, store) -> None:
    rng = np.random.default_rng(args.boot_seed)
    methods = ["mlp_clean+diff", "logreg_clean", "fusion", "mlp_clean+diff +recal", "logreg_clean +recal",
               "fusion +recal", "router"]
    results, boots, picks = {}, {}, {}
    for held_out in GENERATORS:
        seen = [g for g in GENERATORS if g != held_out]
        fit_rec, val_rec = splits(args, seen, seen[0])          # identical to the main LOGO runs
        fit, val = store.gather(fit_rec, "clean", "none"), store.gather(val_rec, "clean", "none")
        dval = {c: store.gather([r for g in seen for r in test_records(args, g)], c, "none") for c in dh.CONDITIONS}
        tests = {c: store.gather(test_records(args, held_out), c, "none") for c in dh.CONDITIONS}
        overlap = {Path(p).name for p in dval["clean"]["path"]} & {Path(p).name for p in tests["clean"]["path"]}
        assert not overlap, f"degraded validation shares {len(overlap)} files with the {held_out} test set"

        t0 = time.time()
        probes = {d: dh.fit_detector(d, fit, args, held_out)[0] for d in ("mlp_clean+diff", "logreg_clean")}

        def scores(emb):
            s = {d: fn(emb) for d, fn in probes.items()}
            s["fusion"] = 0.5 * (logit(s["mlp_clean+diff"]) + logit(s["logreg_clean"]))
            return s
        yv = val["label"].astype(int)
        clean_thr = {k: best_threshold(yv, v)[0] for k, v in scores(val).items()}
        for c in dh.CONDITIONS:
            sd, st = scores(dval[c]), scores(tests[c])
            yd, yt = dval[c]["label"].astype(int), tests[c]["label"].astype(int)
            recal = {k: best_threshold(yd, v) for k, v in sd.items()}
            choice = max(recal, key=lambda k: recal[k][1])        # best on degraded seen generators
            picks[(held_out, c)] = choice
            plan = {**{k: (st[k], clean_thr[k]) for k in st},
                    **{f"{k} +recal": (st[k], recal[k][0]) for k in st},
                    "router": (st[choice], recal[choice][0])}
            boot = Boot(yt, args.bootstrap, rng)
            for m in methods:
                s, t = plan[m]
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


# --------------------------------------------------------------------------- main

def write_report(name: str, lines: list[str], data: dict) -> None:
    text = "\n\n".join(lines) + "\n"
    (ROOT / "results" / f"{name}.md").write_text(text, encoding="utf-8")
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("\n" + text + f"\nSaved -> results/{name}.md and .json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["e3", "e4"])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    # the settings the cached embeddings and the main runs were made with
    defaults = dict(data_root=ROOT / resnet.DEFAULT_DATA_ROOT, test_data_root=ROOT / resnet.DEFAULT_TEST_DATA_ROOT,
                    test_split="test", model_name="dinov2_vitl14", preprocess="crop", lamb=0.05, seed=42,
                    max_per_group=None, C=0.1, hidden=256, device=None, batch_size=32, workers=0)
    for k, v in defaults.items():
        setattr(args, k, v.resolve() if isinstance(v, Path) else v)
    store = make_store(args)
    {"e3": run_e3, "e4": run_e4}[args.experiment](args, store)


if __name__ == "__main__":
    main()
