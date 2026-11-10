"""Extract DINOv2 embeddings of clean and noise-perturbed images for probe training.

For every image in each configured set, saves (as float16):
    clean  - DINOv2 CLS embedding of the preprocessed image
    noisy  - embedding of the same image + lamb * N(0, 1) noise (RIGID's perturbation)
    cos    - cosine similarity between the two (the plain RIGID score)
plus path, label (1 = real, 0 = fake), group (scene id, for leakage-free splits) and
source (generator / dataset name).

Usage:
    uv run python scripts/extract_embeddings.py --config configs/embeddings.yaml [--only SET ...]

One .npz per set in OUTPUT_DIR; existing files are skipped, so it can be resumed.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rigid import RIGIDDetector  # noqa: E402
from utils.datasets import IMAGE_EXTS, ImageListDataset, skip_none_collate  # noqa: E402


def list_images(folder: Path, n: int | None, seed: int) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    if n is not None and n < len(files):
        files = sorted(random.Random(seed).sample(files, n))
    return files


@torch.no_grad()
def extract(detector: RIGIDDetector, files: list[Path], cfg: dict) -> dict:
    ds = ImageListDataset([(p, 0, "") for p in files], detector.transform)
    loader = DataLoader(ds, batch_size=cfg["BATCH_SIZE"], num_workers=cfg["NUM_WORKERS"],
                        collate_fn=skip_none_collate)
    gen = torch.Generator().manual_seed(cfg["SEED"])
    clean, noisy, paths = [], [], []
    for batch in tqdm(loader, leave=False):
        if batch is None:
            continue
        x, _, p = batch
        x = x.to(detector.device)
        noise = torch.randn(x.shape, generator=gen).to(detector.device)
        clean.append(detector.model(x).float().cpu())
        noisy.append(detector.model(x + noise * detector.lamb).float().cpu())
        paths.extend(p)
    clean, noisy = torch.cat(clean), torch.cat(noisy)
    cos = F.cosine_similarity(clean, noisy, dim=-1)
    return {"clean": clean.half().numpy(), "noisy": noisy.half().numpy(),
            "cos": cos.numpy().astype(np.float32), "path": np.array(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="only extract these set names")
    args = parser.parse_args()
    cfg = {"LAMB": 0.05, "MODEL_NAME": "dinov2_vitl14", "IMAGE_SIZE": 224, "BATCH_SIZE": 32,
           "NUM_WORKERS": 4, "DEVICE": None, "SEED": 100, "OUTPUT_DIR": "results/embeddings",
           **yaml.safe_load(args.config.read_text())}

    out_dir = ROOT / cfg["OUTPUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = RIGIDDetector(lamb=cfg["LAMB"], model_name=cfg["MODEL_NAME"], device=cfg["DEVICE"],
                             image_size=cfg["IMAGE_SIZE"], seed=cfg["SEED"])

    for name, spec in cfg["SETS"].items():
        if args.only and name not in args.only:
            continue
        out = out_dir / f"{name}.npz"
        if out.exists():
            print(f"{name}: exists, skipping")
            continue
        files = list_images(Path(spec["path"]).expanduser(), spec.get("n"), cfg["SEED"])
        print(f"{name}: {len(files)} images ({spec['label']}, source={spec['source']})")
        rec = extract(detector, files, cfg)
        n = len(rec["path"])
        # group = scene id: Synthbuster fakes share the RAISE file stem of the photo they were
        # prompted from, so real and fake versions of a scene land in the same split.
        groups = np.array([f"{spec['dataset']}:{Path(p).stem}" if spec.get("group_by_stem")
                           else f"{name}:{Path(p).stem}" for p in rec["path"]])
        np.savez_compressed(out, **rec, group=groups,
                            label=np.full(n, 1 if spec["label"] == "real" else 0, dtype=np.int8),
                            source=np.full(n, spec["source"]), dataset=np.full(n, spec["dataset"]))
        print(f"  saved {n} -> {out.name}  (mean cos {rec['cos'].mean():.4f})")


if __name__ == "__main__":
    main()
