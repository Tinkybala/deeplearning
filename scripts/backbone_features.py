"""Frozen features from several backbones / layers for the GenImage holdout (embedding only).

One decode pass per image feeds two frozen ViT-L/14 backbones:

    DINOv2 ViT-L/14   CLS token and mean patch token after blocks 8, 12, 16, 20, 24
                      (layer-normed, as DINOv2's own output; block 24 CLS = the embedding
                      the main runs use)                -> dino_cls{L}, dino_patch{L}
    CLIP ViT-L/14     OpenAI weights (the UnivFD backbone), projected image embedding
                                                         -> clip

Preprocessing is the ResNet / main-run eval pipeline (shorter side 256, centre crop 224),
then each backbone's own normalisation. Images cover every folder the leave-one-generator-
out analysis needs: the seen-generator training folders (clean) and every generator's
test folders under the six test conditions plus the format-matched condition fmt_r256_q96
(see format_alignment_experiments.py). Seen-generator validation fakes are read from the
test folders, which hold the same files as training/<g>/val/ai.

Features go to results/backbone_embeddings/<folder>/<condition>.npz (git-ignored; one file
per folder and condition, resumable).

    python scripts/backbone_features.py               # everything (~67k images)
    python scripts/backbone_features.py --limit 64    # smoke test on a few images
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import format_alignment_experiments as fa  # noqa: E402  (registers the fmt_* conditions)
from rigid import DEFAULT_MEAN, DEFAULT_STD, get_device, load_model  # noqa: E402

resnet = fa.resnet
GENERATORS = resnet.GENERATORS
FEATURE_ROOT = ROOT / "results" / "backbone_embeddings"
CONDITIONS = fa.dh.CONDITIONS + ["fmt_r256_q96"]
DINO_LAYERS = [8, 12, 16, 20, 24]
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def jobs(args) -> list[tuple[Path, str]]:
    """(folder, condition) pairs, clean training folders first."""
    train = [args.data_root / g / "train" / "ai" for g in GENERATORS]
    train += [args.data_root / g / "train" / "nature" for g in GENERATORS[:2]]   # real photos of seen[0]
    out = [(f, "clean") for f in train]
    for c in CONDITIONS:
        out += [(args.test_data_root / g / "test" / k, c) for g in GENERATORS for k in ("nature", "ai")]
    return out


def cache_path(folder: Path, condition: str) -> Path:
    return FEATURE_ROOT / fa.dh.portable(folder, ROOT / resnet.DEFAULT_TEST_DATA_ROOT) / f"{condition}.npz"


class Backbones:
    def __init__(self, device):
        import open_clip
        self.device = device
        self.dino = load_model("dinov2_vitl14", device=str(device))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            clip, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
        self.clip = clip.visual.eval().requires_grad_(False).to(device)
        self.norm_dino = transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
        self.norm_clip = transforms.Normalize(CLIP_MEAN, CLIP_STD)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        x = x.to(self.device)
        layers = self.dino.get_intermediate_layers(self.norm_dino(x), n=[l - 1 for l in DINO_LAYERS],
                                                   return_class_token=True, norm=True)
        out = {}
        for l, (patch, cls) in zip(DINO_LAYERS, layers):
            out[f"dino_cls{l}"] = cls
            out[f"dino_patch{l}"] = patch.mean(1)
        out["clip"] = self.clip(self.norm_clip(x))
        return {k: v.float().cpu() for k, v in out.items()}


class Items(torch.utils.data.Dataset):
    """(path, condition) items across many folders, so one DataLoader serves the whole run."""

    def __init__(self, items: list[tuple[Path, str]]):
        self.items = items
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, condition = self.items[i]
        return fa.FormatRecords([(path, 0, "")], self.tf, condition)[0][0]


def pending(folder: Path, condition: str, limit: int | None) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.is_file())[:limit]
    path = cache_path(folder, condition)
    if path.exists():
        with np.load(path) as data:
            if set(data["name"]) >= {p.name for p in files}:
                return []
    return files


def save(folder: Path, condition: str, files: list[Path], feats: dict) -> None:
    data = {k: np.concatenate(v).astype(np.float16) for k, v in feats.items()}
    assert all(len(v) == len(files) for v in data.values())
    data["name"] = np.asarray([p.name for p in files])
    path = cache_path(folder, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only the first N files per folder (smoke test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.data_root = (ROOT / resnet.DEFAULT_DATA_ROOT).resolve()
    args.test_data_root = (ROOT / resnet.DEFAULT_TEST_DATA_ROOT).resolve()
    if args.limit:
        global FEATURE_ROOT
        FEATURE_ROOT = FEATURE_ROOT / f"_smoke_limit{args.limit}"
    todo = [(f, c, files) for f, c in jobs(args) if (files := pending(f, c, args.limit))]
    total = sum(len(files) for _, _, files in todo)
    print(f"{len(todo)} folder/condition pairs, {total} images to embed", flush=True)
    if not todo:
        return
    model = Backbones(get_device(args.device))
    items = [(p, c) for _, c, files in todo for p in files]
    loader = DataLoader(Items(items), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    job, buf, feats, done, t0 = 0, 0, {}, 0, time.time()
    for x in loader:
        out = {k: v.numpy() for k, v in model(x).items()}
        pos = 0
        while pos < len(x):                       # split the batch across folder boundaries
            folder, condition, files = todo[job]
            take = min(len(files) - buf, len(x) - pos)
            for k, v in out.items():
                feats.setdefault(k, []).append(v[pos:pos + take])
            pos, buf, done = pos + take, buf + take, done + take
            if buf == len(files):
                save(folder, condition, files, feats)
                rate = done / (time.time() - t0)
                print(f"[{job + 1}/{len(todo)}] {fa.dh.portable(folder, args.test_data_root)} [{condition}]  "
                      f"{done}/{total} images, {rate:.1f} img/s, ~{(total - done) / rate / 60:.0f} min left", flush=True)
                job, buf, feats = job + 1, 0, {}

if __name__ == "__main__":
    main()
