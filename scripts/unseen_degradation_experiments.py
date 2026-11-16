"""Robustness to degradations the detectors never saw in training.

The augmented detectors were trained with JPEG quality 30-95 and downscaling to 32, 64 or 128
pixels, and the six standard test conditions use the same kinds of operation (JPEG Q50, resize
to 128/64/32). So the standard results show robustness to degradations seen in training. This
tests six held-out degradations, outside that training distribution:

    jpeg_q20    JPEG quality 20 (below the trained range 30-95)
    webp_q50    WebP quality 50 (a different codec)
    blur_s1     Gaussian blur, radius 1 pixel, at native resolution (never trained)
    noise_s5    additive Gaussian noise, sigma 5/255, seeded per file (never trained)
    resize_48   downscale the shorter side to 48, back to 256 (a size between the trained 32 and 64)
    resize_96   downscale to 96, back to 256 (between the trained 64 and 128)

Each is applied to the held-out generator's test images (and their real photos) before the
shared preprocessing (shorter side 256, centre crop 224). Embedding (cached, resumable) stores
DINOv2 blocks 20 and 24 (<cond>.npz) and DINOv3 blocks 20 and 24 (<cond>__dinov3.npz) under
results/backbone_embeddings/, about 42,000 images, about 2 h on an M-series Mac.

Detectors (leave-one-generator-out, clean seen-val threshold, as in the main runs):
    final layer (block 24) MLP, clean training     the plain frozen-DINOv2 baseline
    block-20 MLP, clean training                   the layer choice alone
    final detector: block-20 MLP, clean + augmented
    fusion: final detector + DINOv3 block-24 MLP (clean + augmented), logits standardised on
        clean seen-val and averaged
Output: results/unseen_degradation_experiments.{md,json}.

    python scripts/unseen_degradation_experiments.py --check     # pipeline vs cached clean features
    python scripts/unseen_degradation_experiments.py             # embed what is missing, then evaluate
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_comparison as bc  # noqa: E402
import backbone_features as bf  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402
import clip_layer_features as clf  # noqa: E402
from rigid import DEFAULT_MEAN, DEFAULT_STD, get_device, load_model  # noqa: E402

resnet = bf.resnet
GENERATORS, SHORT = cpe.GENERATORS, cpe.SHORT
NEW = ["jpeg_q20", "webp_q50", "blur_s1", "noise_s5", "resize_48", "resize_96"]
LABEL = {"jpeg_q20": "JPEG Q20", "webp_q50": "WebP Q50", "blur_s1": "Blur r1", "noise_s5": "Noise 5/255",
         "resize_48": "Resize 48", "resize_96": "Resize 96",
         "clean": "Clean", "jpeg_q50": "JPEG Q50", "resize_64": "Resize 64", "resize_32": "Resize 32"}
SEEN = ["clean", "jpeg_q50", "resize_64", "resize_32"]      # standard conditions, for reference
AUG = "clean__aug-both"
V3 = "__dinov3"
DETECTORS = ["Final layer (block 24) MLP, clean", "Block-20 MLP, clean",
             "Final detector (block-20 MLP, clean + aug.)", "Fusion with DINOv3 block 24 (clean + aug.)"]
BASE = DETECTORS[2]


def degrade(image: Image.Image, condition: str, name: str) -> Image.Image:
    if condition == "clean":
        return image
    if condition in ("jpeg_q20", "webp_q50"):
        buffer = io.BytesIO()
        fmt, q = ("JPEG", 20) if condition == "jpeg_q20" else ("WEBP", 50)
        image.save(buffer, format=fmt, quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if condition == "blur_s1":
        return image.filter(ImageFilter.GaussianBlur(radius=1))
    if condition == "noise_s5":
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        a = np.asarray(image, dtype=np.float32) + rng.normal(0, 5, (image.height, image.width, 3))
        return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))
    if condition.startswith("resize_"):
        return resnet.resize_shorter_side(resnet.resize_shorter_side(image, int(condition.split("_")[1])), 256)
    raise ValueError(condition)


class Items(Dataset):
    def __init__(self, items):
        self.items = items
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, condition = self.items[i]
        with Image.open(path) as im:
            return self.tf(degrade(im.convert("RGB"), condition, path.name))


class Models:
    def __init__(self, device):
        self.device = device
        self.dino = load_model("dinov2_vitl14", device=str(device))
        self.norm = transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
        self.v3 = clf.DinoV3(device)

    @torch.no_grad()
    def __call__(self, x):
        layers = self.dino.get_intermediate_layers(self.norm(x.to(self.device)), n=[19, 23], return_class_token=True, norm=True)
        d2 = {f"dino_cls{l}": cls.float().cpu().numpy() for l, (_, cls) in zip((20, 24), layers)}
        v3 = self.v3(x)
        return d2, {k: v3[k] for k in ("dinov3_cls20", "dinov3_cls24")}


def folders(args):
    return [args.test_data_root / g / "test" / k for g in GENERATORS for k in ("nature", "ai")]


def save(folder, condition, names, feats):
    data = {k: np.concatenate(v).astype(np.float16) for k, v in feats.items()}
    data["name"] = np.asarray(names)
    path = bf.cache_path(folder, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)


def embed(args, models, conditions, limit=None, only_return=False):
    todo = []
    for c in conditions:
        for f in folders(args):
            files = sorted(p for p in f.iterdir() if p.is_file())[:limit]
            if only_return or not bf.cache_path(f, c + V3).exists():
                todo.append((f, c, files))
    total = sum(len(x[2]) for x in todo)
    print(f"embedding {len(todo)} folder/condition pairs, {total} images", flush=True)
    out, done, t0 = {}, 0, time.time()
    for f, c, files in todo:
        loader = DataLoader(Items([(p, c) for p in files]), batch_size=args.batch_size, num_workers=args.workers)
        f2, f3 = {}, {}
        for x in loader:
            a, b = models(x)
            for k, v in a.items():
                f2.setdefault(k, []).append(v)
            for k, v in b.items():
                f3.setdefault(k, []).append(v)
        names = [p.name for p in files]
        if only_return:
            out[(f, c)] = ({k: np.concatenate(v) for k, v in f2.items()}, {k: np.concatenate(v) for k, v in f3.items()}, names)
        else:
            save(f, c, names, f2)
            save(f, c + V3, names, f3)
        done += len(files)
        rate = done / (time.time() - t0)
        print(f"  {cpe.SHORT.get(f.parent.parent.name, f.parent.parent.name):10s} {f.name:6s} [{c}] {done}/{total}, "
              f"{rate:.1f} img/s, ~{(total - done) / rate / 60:.0f} min left", flush=True)
    return out


def check(args, models):
    """Run the clean condition through this pipeline and compare with the cached features."""
    got = embed(args, models, ["clean"], limit=32, only_return=True)
    worst = {}
    for (f, _), (d2, d3, names) in got.items():
        c2 = np.load(bf.cache_path(f, "clean"))
        c3 = np.load(bf.cache_path(f, "clean" + V3))
        i2 = {n: i for i, n in enumerate(c2["name"])}
        i3 = {n: i for i, n in enumerate(c3["name"])}
        for k in ("dino_cls20", "dino_cls24"):
            worst[k] = max(worst.get(k, 0), float(np.abs(d2[k] - c2[k][[i2[n] for n in names]].astype(np.float32)).max()))
        for k in ("dinov3_cls20", "dinov3_cls24"):
            worst[k] = max(worst.get(k, 0), float(np.abs(d3[k] - c3[k][[i3[n] for n in names]].astype(np.float32)).max()))
    print("max |pipeline - cached| on clean images:", {k: round(v, 4) for k, v in worst.items()})


def evaluate(args):
    fstore = bc.FeatureStore()
    rng = np.random.default_rng(args.boot_seed)
    conds = SEEN + NEW
    res, boots = {}, {}
    for g in GENERATORS:
        seen = [x for x in GENERATORS if x != g]
        fit_rec, val_rec = cpe.splits(args, seen, seen[0])
        val_new = bc.fa.shared_val(args, val_rec)
        test_rec = cpe.test_records(args, g)
        yf, yv, yt = (np.asarray([r[1] for r in rec]) for rec in (fit_rec, val_rec, test_rec))
        boot = {c: cpe.Boot(yt, args.bootstrap, rng) for c in conds}

        def probe(key, suffix, aug):
            X, y = fstore.gather(fit_rec, "clean" + suffix, [key]), yf
            if aug:
                X, y = np.vstack([X, fstore.gather(fit_rec, AUG + suffix, [key])]), np.concatenate([yf, yf])
            fn = bc.fit_probe("MLP", X, y, args)
            return {"val": cpe.logit(fn(fstore.gather(val_new, "clean" + suffix, [key])))} | {
                c: cpe.logit(fn(fstore.gather(test_rec, c + suffix, [key]))) for c in conds}

        s = {DETECTORS[0]: probe("dino_cls24", "", False), DETECTORS[1]: probe("dino_cls20", "", False),
             DETECTORS[2]: probe("dino_cls20", "", True)}
        v3 = probe("dinov3_cls24", V3, True)
        z = lambda d, k: (d[k] - d["val"].mean()) / d["val"].std()  # noqa: E731
        s[DETECTORS[3]] = {k: 0.5 * (z(s[DETECTORS[2]], k) + z(v3, k)) for k in ["val"] + conds}
        for m, d in s.items():
            thr = cpe.best_threshold(yv, d["val"])[0]
            for c in conds:
                res[(g, m, c)] = cpe.metrics(yt, d[c], thr)
                boots[(g, m, c)] = boot[c](d[c], thr)
        print(f"{SHORT[g]:10s} final detector: " + "  ".join(f"{LABEL[c]} {res[(g, BASE, c)]['balanced_accuracy']:.3f}"
                                                         for c in conds), flush=True)
    report(args, res, boots)


def report(args, res, boots):
    conds = SEEN + NEW
    gens = GENERATORS

    def mean(m, c, metric="balanced_accuracy"):
        return float(np.mean([res[(g, m, c)][metric] for g in gens]))

    def mean_new(m, metric="balanced_accuracy"):
        return float(np.mean([mean(m, c, metric) for c in NEW]))

    lines = ["# Held-out degradations: robustness to transformations never seen in training",
             "Training augmentation (the *aug.* detectors) used JPEG quality 30-95 and downscaling to 32, 64 or 128 px; "
             "the four **seen-type** conditions use the same operations. The six **held-out** conditions are outside "
             "that training distribution: JPEG Q20 (below the trained range), WebP Q50 (another codec), Gaussian blur "
             "(radius 1 px, native resolution), additive Gaussian noise (sigma 5/255), and downscaling to 48 or 96 px "
             "(sizes between the trained ones). Leave-one-generator-out, clean seen-val threshold, mean over the 7 holdouts."]
    for metric, title, f in (("balanced_accuracy", "Balanced accuracy", lambda x: f"{x:.1%}"),
                             ("auroc_real_positive", "AUROC", lambda x: f"{x:.3f}")):
        rows = ["| Detector | " + " | ".join(LABEL[c] for c in conds) + " | Mean, held-out |",
                "|---|" + "---:|" * (len(conds) + 1)]
        for m in DETECTORS:
            rows.append(f"| {m} | " + " | ".join(f(mean(m, c, metric)) for c in conds) + f" | {f(mean_new(m, metric))} |")
        lines += [f"## {title}", "\n".join(rows)]
    rows = ["| Comparison | " + " | ".join(LABEL[c] for c in conds) + " | Mean, held-out |", "|---|" + "---:|" * (len(conds) + 1)]
    for a, b in ((BASE, DETECTORS[0]), (BASE, DETECTORS[1]), (DETECTORS[3], BASE)):
        cells = []
        for c in conds:
            point = np.mean([res[(g, a, c)]["balanced_accuracy"] - res[(g, b, c)]["balanced_accuracy"] for g in gens])
            rep = np.mean([boots[(g, a, c)]["balanced_accuracy"] - boots[(g, b, c)]["balanced_accuracy"] for g in gens], axis=0)
            cells.append(cpe.fmt_diff(point, rep, 100, 1))
        point = np.mean([[res[(g, a, c)]["balanced_accuracy"] - res[(g, b, c)]["balanced_accuracy"] for g in gens] for c in NEW])
        rep = np.mean([np.mean([boots[(g, a, c)]["balanced_accuracy"] - boots[(g, b, c)]["balanced_accuracy"] for g in gens], axis=0)
                       for c in NEW], axis=0)
        cells.append(cpe.fmt_diff(point, rep, 100, 1))
        rows.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += [f"## Differences, balanced accuracy (pp), paired bootstrap ({args.bootstrap} replicates per holdout); "
              "* = 95% interval excludes 0", "\n".join(rows)]
    rows = ["| Detector | " + " | ".join(LABEL[c] for c in conds) + " |", "|---|" + "---:|" * len(conds)]
    for m in DETECTORS:
        rows.append(f"| {m} | " + " | ".join(f"{mean(m, c, 'real_recall'):.0%} / {mean(m, c, 'generated_recall'):.0%}"
                                             for c in conds) + " |")
    lines += ["## Class recall (real / generated)", "\n".join(rows)]
    rows = ["| Held-out generator | " + " | ".join(LABEL[c] for c in NEW) + " |", "|---|" + "---:|" * len(NEW)]
    for g in gens:
        rows.append(f"| {SHORT[g]} | " + " | ".join(f"{res[(g, BASE, c)]['balanced_accuracy']:.1%}" for c in NEW) + " |")
    lines += ["## Final detector per held-out generator, held-out conditions", "\n".join(rows)]
    cpe.write_report("unseen_degradation_experiments", lines, {f"{g}|{m}|{c}": x for (g, m, c), x in res.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare the pipeline with cached clean features and exit")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()
    cpe.set_defaults(args)
    if args.check or not args.no_embed:
        models = Models(get_device(None))
        if args.check:
            return check(args, models)
        embed(args, models, NEW)
        del models
    evaluate(args)


if __name__ == "__main__":
    main()
