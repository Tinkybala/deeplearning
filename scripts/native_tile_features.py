"""DINOv2 features at native resolution: 224 px tiles instead of one downscaled view.

Every detector so far sees the image resized to shorter side 256 and centre-cropped to 224,
which throws away fine detail in large images (SD v1.5 and Wukong are 512 px, Midjourney
1024 px, real photos about 375 x 500). Here each image keeps its pixels:

    crop            the main runs' view (shorter side 256, centre 224), as the baseline
    native centre   the central 224 x 224 crop at native resolution
    tile mean       mean CLS token over a grid of 224 x 224 tiles at native resolution:
                    2 positions along a side of at least 448 px (the centres of its two
                    halves), else 1; up to 4 tiles
Images with a side under 224 px are first upscaled to shorter side 224 (BigGAN is 128 px).

Folders: the clean training folders and every test folder under clean, JPEG Q50 and Format
q96 (the conditions in which the image keeps its size; after the resize conditions it is at
most 256 px, where tiling is just a centre crop). DINOv2 ViT-L/14, layer-normed CLS after
blocks 20 and 24. Output: results/backbone_embeddings/<folder>/<condition>__tiles.npz (keys
dino_crop{20,24}, dino_center{20,24}, dino_tilemean{20,24}, n_tiles; git-ignored, resumable). About 60k images.

    python scripts/native_tile_features.py [--limit 32]
"""

from __future__ import annotations

import argparse
import sys
import time
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

fa = bf.fa
dh = fa.dh
resnet = fa.resnet
LAYERS = [20, 24]
SUFFIX = "__tiles"
TEST_CONDS = ["clean", "jpeg_q50", "fmt_q96"]
NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(bf.DEFAULT_MEAN, bf.DEFAULT_STD)])
CROP = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), NORM])   # the main runs' view


def positions(n: int) -> list[int]:
    return [(n - 224) // 2] if n < 448 else [max(0, min(n - 224, n // 4 - 112)), max(0, min(n - 224, 3 * n // 4 - 112))]


def tiles(image) -> torch.Tensor:
    """(2 + k, 3, 224, 224): the main runs' downscaled crop, the native centre crop, then the k
    grid tiles (k = 1 reuses the centre)."""
    crop = CROP(image)
    if min(image.size) < 224:
        image = resnet.resize_shorter_side(image, 224)
    w, h = image.size
    crops = [image.crop(((w - 224) // 2, (h - 224) // 2, (w - 224) // 2 + 224, (h - 224) // 2 + 224))]
    grid = [(x, y) for y in positions(h) for x in positions(w)]
    if len(grid) > 1:
        crops += [image.crop((x, y, x + 224, y + 224)) for x, y in grid]
    return torch.stack([crop] + [NORM(c) for c in crops])


class Items(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, condition = self.items[i]
        return fa.FormatRecords([(path, 0, "")], tiles, condition)[0][0]


def collate(batch):
    return torch.cat(batch), torch.tensor([len(b) for b in batch])


def jobs(args):
    train = [args.data_root / g / "train" / "ai" for g in bf.GENERATORS]
    train += [args.data_root / g / "train" / "nature" for g in bf.GENERATORS[:2]]
    out = [(f, "clean") for f in train]
    for c in TEST_CONDS:
        out += [(args.test_data_root / g / "test" / k, c) for g in bf.GENERATORS for k in ("nature", "ai")]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="images per batch (each 1-5 tiles)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cpe.set_defaults(args)
    if args.limit:
        bf.FEATURE_ROOT = bf.FEATURE_ROOT / f"_smoke_tiles_limit{args.limit}"
    todo = []
    for folder, c in jobs(args):
        if not bf.cache_path(folder, c + SUFFIX).exists():
            todo.append((folder, c, sorted(p for p in folder.iterdir() if p.is_file())[:args.limit]))
    total = sum(len(f) for _, _, f in todo)
    print(f"{len(todo)} folder/condition pairs, {total} images", flush=True)
    if not todo:
        return
    device = bf.get_device(None)
    dino = bf.load_model("dinov2_vitl14", device=str(device))
    items = [(p, c) for _, c, files in todo for p in files]
    loader = DataLoader(Items(items), batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate)
    job, buf, feats, done, t0, ntiles = 0, 0, {}, 0, time.time(), 0
    with torch.no_grad():
        for x, counts in loader:
            layers = dino.get_intermediate_layers(x.to(device), n=[l - 1 for l in LAYERS], return_class_token=True, norm=True)
            cls = {l: c.float().cpu() for l, (_, c) in zip(LAYERS, layers)}
            ntiles += len(x)
            starts = torch.cumsum(counts, 0) - counts
            per = {}
            for l in LAYERS:
                per[f"dino_crop{l}"] = torch.stack([cls[l][s] for s in starts])
                per[f"dino_center{l}"] = torch.stack([cls[l][s + 1] for s in starts])
                per[f"dino_tilemean{l}"] = torch.stack([cls[l][s + (2 if n > 2 else 1): s + n].mean(0)
                                                        for s, n in zip(starts.tolist(), counts.tolist())])
            per["n_tiles"] = torch.clamp(counts - 2, min=1).float()[:, None]
            per = {k: v.numpy() for k, v in per.items()}
            pos = 0
            while pos < len(counts):
                folder, c, files = todo[job]
                take = min(len(files) - buf, len(counts) - pos)
                for k, v in per.items():
                    feats.setdefault(k, []).append(v[pos:pos + take])
                pos, buf, done = pos + take, buf + take, done + take
                if buf == len(files):
                    data = {k: np.concatenate(v).astype(np.float16) for k, v in feats.items()}
                    data["name"] = np.asarray([p.name for p in files])
                    path = bf.cache_path(folder, c + SUFFIX)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(path.stem + ".tmp.npz")
                    np.savez(tmp, **data)
                    tmp.replace(path)
                    rate = done / (time.time() - t0)
                    print(f"[{job + 1}/{len(todo)}] {dh.portable(folder, args.test_data_root)} [{c}]  {done}/{total}, "
                          f"{rate:.1f} img/s ({ntiles / done:.1f} tiles/img), ~{(total - done) / rate / 60:.0f} min left",
                          flush=True)
                    job, buf, feats = job + 1, 0, {}


if __name__ == "__main__":
    main()
