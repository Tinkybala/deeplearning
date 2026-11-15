"""Score every test image with Tyrone's seven leave-one-generator-out ResNet-18 checkpoints.

The checkpoints (`best_model.pt` from `train_resnet_generalization.py --eval-format squash224`,
run names resnet_rigid_format_holdout_<generator>) are not in git; pass their folder with
--checkpoints. Each image is decoded and degraded once and then scored by all seven models
under two eval preprocessings:

    squash224     Resize((224, 224)), the preprocessing of Tyrone's published ResNet results
    center_crop   Resize(256) + CenterCrop(224), the main DINOv2 runs' preprocessing

Images: every generator's test folders (real and generated) under the six test conditions and
the two format-matched conditions of format_alignment_experiments.py, plus the clean
calibration real photos (20% of seen[0]'s train/nature) that the validation split uses. A
generator's test folder serves as its own held-out test set and as degraded seen-generator
validation for the other six holdouts; the seen-generator validation fakes are the same files
as the test fakes (training/<g>/val/ai == test/<g>/test/ai).

Scores are the logit margin log p(real) - log p(generated) (monotone in the checkpoints'
p_real, so thresholds carry over). Output: results/resnet_scores/<folder>/<condition>.npz,
float32 of shape (images, 7 checkpoints, 2 preprocessings), resumable.

    python scripts/resnet_checkpoint_scores.py --checkpoints "<...>/Frozen Resnet model/checkpoints"
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import format_alignment_experiments as fa  # noqa: E402  (registers the fmt_* conditions)
from rigid import get_device  # noqa: E402

resnet = fa.resnet
GENERATORS = resnet.GENERATORS
SCORE_ROOT = ROOT / "results" / "resnet_scores"
CONDITIONS = fa.dh.CONDITIONS + list(fa.FORMAT)
PREPROC = ("squash224", "center_crop")
SEED = 42


def run_name(g: str) -> str:
    return f"resnet_rigid_format_holdout_{g}"


def calibration_reals(data_root: Path, real_source: str) -> list[Path]:
    """The 20% of real_source's train/nature that train_resnet_generalization.py validates on."""
    real = resnet.files_for(data_root, real_source, "train", "nature")
    random.Random(SEED).shuffle(real)
    return sorted(real[:max(1, len(real) // 5)])


def jobs(args) -> list[tuple[Path, str, list[Path]]]:
    """(folder, condition, files): calibration reals first, then all test folders."""
    out = []
    for src in GENERATORS[:2]:            # seen[0] is BigGAN, or VQDM when BigGAN is held out
        out.append((args.data_root / src / "train" / "nature", "clean", calibration_reals(args.data_root, src)))
    for c in CONDITIONS:
        for g in GENERATORS:
            for k in ("nature", "ai"):
                folder = args.test_data_root / g / "test" / k
                out.append((folder, c, sorted(p for p in folder.iterdir() if p.is_file())))
    return out


def cache_path(folder: Path, condition: str) -> Path:
    return SCORE_ROOT / fa.dh.portable(folder, ROOT / resnet.DEFAULT_TEST_DATA_ROOT) / f"{condition}.npz"


class TwoViews:
    def __init__(self):
        norm = [transforms.ToTensor(), transforms.Normalize(ResNet18_Weights.DEFAULT.transforms().mean,
                                                            ResNet18_Weights.DEFAULT.transforms().std)]
        self.views = [transforms.Compose([transforms.Resize((224, 224))] + norm),
                      transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)] + norm)]

    def __call__(self, image):
        return torch.stack([v(image) for v in self.views])


class Items(torch.utils.data.Dataset):
    def __init__(self, items: list[tuple[Path, str]]):
        self.items, self.tf = items, TwoViews()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, condition = self.items[i]
        return fa.FormatRecords([(path, 0, "")], self.tf, condition)[0][0]


def load_models(ckpt_dir: Path, device) -> list[torch.nn.Module]:
    models = []
    for g in GENERATORS:
        ck = torch.load(ckpt_dir / run_name(g) / "best_model.pt", map_location="cpu", weights_only=True)
        assert ck["held_out"] == g and ck["seed"] == SEED, (g, ck["held_out"], ck["seed"])
        m = resnet.make_model(False, torch.device("cpu"))
        m.load_state_dict(ck["model_state_dict"])
        models.append(m.eval().to(device))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="folder holding resnet_rigid_format_holdout_<generator>/best_model.pt")
    parser.add_argument("--limit", type=int, default=None, help="only the first N files per folder (smoke test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.data_root = (ROOT / resnet.DEFAULT_DATA_ROOT).resolve()
    args.test_data_root = (ROOT / resnet.DEFAULT_TEST_DATA_ROOT).resolve()
    if args.limit:
        global SCORE_ROOT
        SCORE_ROOT = SCORE_ROOT / f"_smoke_limit{args.limit}"
    todo = [(f, c, files[:args.limit]) for f, c, files in jobs(args) if not cache_path(f, c).exists()]
    total = sum(len(files) for _, _, files in todo)
    print(f"{len(todo)} folder/condition pairs, {total} images to score", flush=True)
    if not todo:
        return
    device = get_device(args.device)
    models = load_models(args.checkpoints, device)
    items = [(p, c) for _, c, files in todo for p in files]
    loader = DataLoader(Items(items), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    job, buf, done, t0 = 0, [], 0, time.time()
    with torch.inference_mode():
        for x in loader:                                   # x: (B, 2 views, 3, 224, 224)
            b = len(x)
            flat = x.flatten(0, 1).to(device)
            out = torch.stack([(lambda z: z[:, 1] - z[:, 0])(m(flat).float()) for m in models], 1)
            scores = out.view(b, 2, len(models)).transpose(1, 2).cpu().numpy()   # (B, 7, 2)
            pos = 0
            while pos < b:                                 # split the batch across folder boundaries
                folder, condition, files = todo[job]
                take = min(len(files) - sum(len(s) for s in buf), b - pos)
                buf.append(scores[pos:pos + take])
                pos += take
                done += take
                if sum(len(s) for s in buf) == len(files):
                    path = cache_path(folder, condition)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(path.stem + ".tmp.npz")
                    np.savez(tmp, score=np.concatenate(buf).astype(np.float32),
                             name=np.asarray([p.name for p in files]),
                             generators=np.asarray(GENERATORS), preproc=np.asarray(PREPROC))
                    tmp.replace(path)
                    rate = done / (time.time() - t0)
                    print(f"[{job + 1}/{len(todo)}] {fa.dh.portable(folder, args.test_data_root)} [{condition}]  "
                          f"{done}/{total}, {rate:.1f} img/s, ~{(total - done) / rate / 60:.0f} min left", flush=True)
                    job, buf = job + 1, []


if __name__ == "__main__":
    main()
