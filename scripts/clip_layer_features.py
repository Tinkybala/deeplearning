"""Intermediate-block CLIP ViT-L/14 (or DINOv3 ViT-L/16) features for the GenImage holdout (embedding only).

backbone_comparison.py found CLIP's final embedding strong on clean images but fragile under
JPEG and resizing, as DINOv2's final layer is, while DINOv2's intermediate blocks were both
strong and (with augmented training) robust. This stores the CLS token of the residual stream
after CLIP blocks 8, 12, 16, 20 and 24 (OpenAI ViT-L/14 weights, the UnivFD backbone), for:

    the clean training folders (7 generators' train/ai, BigGAN's and VQDM's train/nature)
    exactly the augmented ("both") training images of augmented_backbone_features.py: same
        files, seeds and DataLoader worker streams; the worker count each folder was cached
        with is detected on its first batches against the cached DINOv2 embedding
    every test folder under the six test conditions and Format 256+q96

Preprocessing is the main runs' (shorter side 256, centre crop 224), then CLIP's own
normalisation. Output: results/backbone_embeddings/<folder>/<condition>__clipmid.npz with
keys clip_cls8 ... clip_cls24 (git-ignored, resumable). About 130k images, 2-3 h.

--backbone dinov3 stores the same blocks of DINOv3 ViT-L/16 (timm's copy of Meta's LVD-1689M
weights, vit_large_patch16_dinov3.lvd1689m; DINOv3 License), layer-normed by the final norm as
the DINOv2 features are, with ImageNet normalisation: <condition>__dinov3.npz, keys
dinov3_cls8 ... dinov3_cls24.

    python scripts/clip_layer_features.py [--limit 64]
    python scripts/clip_layer_features.py --backbone dinov3
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_features as bf  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402

dh = bf.fa.dh
LAYERS = [8, 12, 16, 20, 24]
SUFFIX = "__clipmid"
SUFFIXES = {"clip": "__clipmid", "dinov3": "__dinov3"}
AUG = "clean__aug-both"


class Clip:
    def __init__(self, device):
        import open_clip
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
        self.visual = model.visual.eval().requires_grad_(False).to(device)
        self.device = device
        self.norm = transforms.Normalize(bf.CLIP_MEAN, bf.CLIP_STD)
        self.out = {}
        for l in LAYERS:
            self.visual.transformer.resblocks[l - 1].register_forward_hook(
                lambda mod, inp, o, l=l: self.out.__setitem__(l, o[:, 0]))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        self.out.clear()
        self.visual(self.norm(x.to(self.device)))
        return {f"clip_cls{l}": self.out[l].float().cpu().numpy() for l in LAYERS}


class DinoV3:
    """DINOv3 ViT-L/16 through timm; CLS token after each block in LAYERS, through the final norm."""
    NAME = "vit_large_patch16_dinov3.lvd1689m"

    def __init__(self, device):
        import timm
        self.model = timm.create_model(self.NAME, pretrained=True).eval().requires_grad_(False).to(device)
        self.device = device
        cfg = self.model.pretrained_cfg
        self.norm = transforms.Normalize(cfg["mean"], cfg["std"])
        self.out = {}
        for l in LAYERS:
            self.model.blocks[l - 1].register_forward_hook(
                lambda mod, inp, o, l=l: self.out.__setitem__(l, o[:, 0]))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        self.out.clear()
        self.model.forward_features(self.norm(x.to(self.device)))
        return {f"dinov3_cls{l}": self.model.norm(self.out[l]).float().cpu().numpy() for l in LAYERS}


def save(path: Path, names: list[str], feats: dict) -> None:
    data = {k: np.concatenate(v).astype(np.float16) for k, v in feats.items()}
    assert all(len(v) == len(names) for v in data.values())
    data["name"] = np.asarray(names)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)


def run_plain(args, model, limit):
    """Clean training folders and all test folders, through one DataLoader (as backbone_features)."""
    jobs = [(f, c) for f, c in bf.jobs(args)]
    todo = []
    for folder, c in jobs:
        files = sorted(p for p in folder.iterdir() if p.is_file())[:limit]
        if not bf.cache_path(folder, c + SUFFIX).exists():
            todo.append((folder, c, files))
    total = sum(len(f) for _, _, f in todo)
    print(f"plain: {len(todo)} folder/condition pairs, {total} images", flush=True)
    if not todo:
        return
    items = [(p, c) for _, c, files in todo for p in files]
    loader = DataLoader(bf.Items(items), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    job, buf, feats, done, t0 = 0, 0, {}, 0, time.time()
    for x in loader:
        out = model(x)
        pos = 0
        while pos < len(x):
            folder, c, files = todo[job]
            take = min(len(files) - buf, len(x) - pos)
            for k, v in out.items():
                feats.setdefault(k, []).append(v[pos:pos + take])
            pos, buf, done = pos + take, buf + take, done + take
            if buf == len(files):
                save(bf.cache_path(folder, c + SUFFIX), [p.name for p in files], feats)
                rate = done / (time.time() - t0)
                print(f"[{job + 1}/{len(todo)}] {dh.portable(folder, args.test_data_root)} [{c}]  {done}/{total}, "
                      f"{rate:.1f} img/s, ~{(total - done) / rate / 60:.0f} min left", flush=True)
                job, buf, feats = job + 1, 0, {}


def run_augmented(args, model, limit):
    """The augmented training images, reproduced exactly (see augmented_backbone_features.py)."""
    store = dh.EmbeddingStore(args)
    dino = None
    folders = [args.data_root / g / "train" / "ai" for g in bf.GENERATORS]
    folders += [args.data_root / g / "train" / "nature" for g in bf.GENERATORS[:2]]
    for folder in folders:
        rel = dh.portable(folder, args.test_data_root)
        out = bf.cache_path(folder, AUG + SUFFIX)
        if out.exists():
            continue
        ref = np.load(store.root / rel / f"{AUG}.npz")
        names = list(ref["name"])[:limit]
        seed = (args.seed * 1_000_003 + zlib.crc32(rel.encode())) % 2**31
        # the DINOv2 pipeline minus its normalisation: the random stream is unchanged
        tf = dh.build_transform(args.preprocess, "both")
        tf = transforms.Compose([t for t in tf.transforms if not isinstance(t, transforms.Normalize)])
        dataset = dh.resnet.ImageRecords([(folder / n, 0, "") for n in names], tf, "clean")

        def loader(workers):
            random.seed(seed)
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                              generator=torch.Generator().manual_seed(seed))

        if dino is None:
            dino = bf.load_model("dinov2_vitl14", device=str(model.device))
            dnorm = transforms.Normalize(bf.DEFAULT_MEAN, bf.DEFAULT_STD)

        def matches(workers):
            got = []
            with torch.no_grad():
                for i, (x, _, _, _) in enumerate(loader(workers)):
                    if i >= 4:
                        break
                    got.append(dino(dnorm(x).to(model.device)).float().cpu().numpy())
            got = np.concatenate(got)
            return np.abs(got - ref["clean"][:len(got)].astype(np.float32)).max() < 0.05

        workers = next((w for w in (4, 0) if matches(w)), None)
        assert workers is not None, f"{rel}: cannot reproduce the cached augmented images"
        feats, t0 = {}, time.time()
        for x, _, _, _ in loader(workers):
            for k, v in model(x).items():
                feats.setdefault(k, []).append(v)
        save(out, names, feats)
        print(f"aug: {rel}: {len(names)} images ({workers} workers), {len(names) / (time.time() - t0):.1f} img/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only the first N files per folder (smoke test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone", choices=list(SUFFIXES), default="clip")
    args = parser.parse_args()
    cpe.set_defaults(args)
    global SUFFIX
    SUFFIX = SUFFIXES[args.backbone]
    if args.limit:
        bf.FEATURE_ROOT = bf.FEATURE_ROOT / f"_smoke_{args.backbone}_limit{args.limit}"
    model = (Clip if args.backbone == "clip" else DinoV3)(bf.get_device(None))
    run_augmented(args, model, args.limit)
    run_plain(args, model, args.limit)


if __name__ == "__main__":
    main()
