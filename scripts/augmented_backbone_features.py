"""DINOv2 intermediate-layer features of the augmented ("both") training images.

augmented_probe_experiments.py fits probes on the final-layer DINOv2 embeddings of the seen
generators' training images after random JPEG / resize artifacts (EmbeddingStore,
augmentation "both"). This recomputes exactly the same augmented images - same files, same
order, same seed, same DataLoader worker random streams - and stores the CLS and mean patch
tokens after blocks 8, 12, 16, 20 and 24 like backbone_features.py, so block-20 probes can
be fit on augmented images too. As a check, the block-24 CLS must equal the cached
augmented embedding ("clean" field of clean__aug-both.npz) to float16 precision.

Output: results/backbone_embeddings/<folder>/clean__aug-both.npz (git-ignored). About 17,600
images, roughly 30 min on an M-series Mac.

    python scripts/augmented_backbone_features.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_features as bf  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402

dh = bf.fa.dh
AUG = "both"
CONDITION = f"clean__aug-{AUG}"


def folders(args) -> list[Path]:
    out = [args.data_root / g / "train" / "ai" for g in bf.GENERATORS]
    return out + [args.data_root / g / "train" / "nature" for g in bf.GENERATORS[:2]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    cpe.set_defaults(args)
    store = dh.EmbeddingStore(args)
    model = None
    for folder in folders(args):
        rel = dh.portable(folder, args.test_data_root)
        out = bf.cache_path(folder, CONDITION)
        if out.exists():
            print(f"{rel}: done")
            continue
        ref = np.load(store.root / rel / f"{CONDITION}.npz")
        names = list(ref["name"])
        assert names == sorted(names), f"{rel}: cache was built in several passes; seeds would not match"
        if model is None:
            model = bf.Backbones.__new__(bf.Backbones)      # DINOv2 only (no CLIP pass)
            model.device = bf.get_device(None)
            model.dino = bf.load_model("dinov2_vitl14", device=str(model.device))
        # identical to EmbeddingStore._embed for a folder embedded in one pass. The random
        # artifacts depend on the DataLoader worker count it used (4 by default in
        # dinov2_genimage_holdout.py, 0 in the cached-probe scripts): try both on two batches.
        seed = (args.seed * 1_000_003 + zlib.crc32(rel.encode())) % 2**31
        dataset = dh.resnet.ImageRecords([(folder / n, 0, "") for n in names], dh.build_transform(args.preprocess, AUG), "clean")

        def run(workers, max_batches=None):
            random.seed(seed)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                                generator=torch.Generator().manual_seed(seed))
            feats = {}
            with torch.no_grad():
                for i, (x, _, _, _) in enumerate(loader):
                    if max_batches is not None and i >= max_batches:
                        break
                    layers = model.dino.get_intermediate_layers(x.to(model.device), n=[l - 1 for l in bf.DINO_LAYERS],
                                                                return_class_token=True, norm=True)
                    for l, (patch, cls) in zip(bf.DINO_LAYERS, layers):
                        feats.setdefault(f"dino_cls{l}", []).append(cls.float().cpu().numpy())
                        feats.setdefault(f"dino_patch{l}", []).append(patch.mean(1).float().cpu().numpy())
            return feats

        def mismatch(feats):
            got = np.concatenate(feats["dino_cls24"])
            return float(np.abs(got - ref["clean"][:len(got)].astype(np.float32)).max())

        workers = next((w for w in (4, 0) if mismatch(run(w, 8)) < 0.05), None)
        assert workers is not None, f"{rel}: neither 4 nor 0 workers reproduces the cached augmented images"
        t0 = time.time()
        feats = run(workers)
        data = {k: np.concatenate(v).astype(np.float16) for k, v in feats.items()}
        diff = np.abs(data["dino_cls24"].astype(np.float32) - ref["clean"].astype(np.float32)).max()
        assert diff < 0.05, f"{rel}: block-24 CLS differs from the cached augmented embedding by {diff}"
        data["name"] = np.asarray(names)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **data)
        tmp.replace(out)
        print(f"{rel}: {len(names)} images ({workers} workers), {len(names) / (time.time() - t0):.1f} img/s, "
              f"block-24 CLS matches the cached augmented embedding (max |diff| {diff:.4f})", flush=True)


if __name__ == "__main__":
    main()
