"""Dataset registry for RIGID evaluation.

A dataset is a DatasetSpec: one or more *parts* holding real images and one or more
holding AI-generated images. Each part says where its files come from (a source in
utils/sources.py plus a reference such as a Kaggle handle) and which folders to scan.
Real and fake images can therefore come from different places (e.g. Synthbuster fakes
from Kaggle + RAISE-1k reals from a local folder).

`load_dataset(name, ...)` returns a torch Dataset yielding (image_tensor, label, path),
label 1 = real, 0 = AI-generated. Each item also has a `group` (the folder it came
from, e.g. "dalle3"), used for per-generator results.

To add a dataset, add a DatasetSpec to DATASETS.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, default_collate

from .sources import fetch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL, FAKE = 1, 0

# Real camera photos (e.g. RAISE TIFFs, ~12 MP) exceed PIL's default bomb-check limit
# on some cameras; they are trusted dataset files, so allow larger images.
Image.MAX_IMAGE_PIXELS = 200_000_000


@dataclass
class Part:
    source: str            # key in utils.sources.SOURCES
    ref: str               # Kaggle handle / local path ("" = must come from the config)
    dirs: list[str]        # folder globs relative to the part's root; "{split}" is filled in


@dataclass
class DatasetSpec:
    real: list[Part]
    fake: list[Part]
    # split name -> value substituted for "{split}" in the folder globs
    splits: dict[str, str] = field(default_factory=lambda: {"all": "*"})
    description: str = ""


SYNTHBUSTER_GENERATORS = [
    "dalle2", "dalle3", "firefly", "glide", "midjourney-v5",
    "stable-diffusion-1-3", "stable-diffusion-1-4", "stable-diffusion-2", "stable-diffusion-xl",
]

DATASETS: dict[str, DatasetSpec] = {
    "cifake": DatasetSpec(
        real=[Part("local", "", ["{split}/REAL"])],
        fake=[Part("local", "", ["{split}/FAKE"])],
        splits={"test": "test/CiFake/test", "train": "training/CiFake/train"},
        description="CIFAKE: CIFAR-10 real vs Stable Diffusion v1.4 fakes, 32x32. "
                    "Expected under dataset/image/{training/CiFake/train,test/CiFake/test}.",
    ),
    "synthbuster": DatasetSpec(
        # RAISE-1k is not redistributable: download with scripts/download_raise1k.py and
        # set REAL_DATA_ROOT to that folder.
        real=[Part("local", "", ["."])],
        fake=[Part("kaggle", "maryjoy12/synthbuster", ["synthbuster/{split}"])],
        splits={"all": "*", **{g: g for g in SYNTHBUSTER_GENERATORS}},
        description="Synthbuster: 9 generators x 1000 PNGs (fake) vs RAISE-1k camera TIFFs (real). "
                    "Split = one generator, or 'all'.",
    ),
    # RIGID paper benchmark (dgm-eval ImageNet256 images from the IBM README's Google Drive).
    # Fetch a subset with scripts/fetch_zip_subset.py into <DATA_ROOT>/Imagenet and
    # <DATA_ROOT>/Imagenet256-<generator>, then set DATA_ROOT.
    "rigid_imagenet": DatasetSpec(
        real=[Part("local", "", ["Imagenet"])],
        fake=[Part("local", "", ["{split}"])],
        splits={"all": "Imagenet256-*", "adm": "Imagenet256-ADM"},
        description="ImageNet256 real (val) vs generated sets from the RIGID paper; "
                    "add a split per generator folder you fetch.",
    ),
    # Generic: any local folder with real/ and fake/ subfolders (set DATA_ROOT in the config).
    "real_fake_folder": DatasetSpec(
        real=[Part("local", "", ["real"])],
        fake=[Part("local", "", ["fake"])],
        description="<DATA_ROOT>/real/** and <DATA_ROOT>/fake/**",
    ),
}


class ImageListDataset(Dataset):
    """Items are (path, label, group)."""

    def __init__(self, items: list[tuple[Path, int, str]], transform: Callable | None = None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        """Returns None for an unreadable image (use `skip_none_collate` in the DataLoader)."""
        path, label, _ = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, SyntaxError) as e:  # truncated / corrupt file
            warnings.warn(f"Skipping unreadable image {path}: {e}")
            return None
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(path)

    @property
    def labels(self) -> list[int]:
        return [label for _, label, _ in self.items]

    @property
    def groups(self) -> list[str]:
        return [group for _, _, group in self.items]


def skip_none_collate(batch):
    """DataLoader collate_fn that drops items the dataset returned as None."""
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


def _scan(root: Path, patterns: list[str], split_value: str, default_group: str) -> list[tuple[Path, str]]:
    """All images under the folders matching `patterns`, with a group name per image."""
    found: list[tuple[Path, str]] = []
    for pat in patterns:
        uses_split = "{split}" in pat
        pattern = pat.format(split=split_value)
        dirs = [root] if pattern in ("", ".") else sorted(root.glob(pattern))
        for d in dirs:
            if not d.is_dir():
                continue
            # Group by folder name when the glob picks the folder (e.g. per generator).
            group = d.name if uses_split and any(c in split_value for c in "*?[") else default_group
            found.extend((p, group) for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTS)
    return found


def _collect(parts: list[Part], root_override: str | None, split_value: str,
             default_group: str, cache_dir: str | None) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for part in parts:
        root = fetch("local", root_override) if root_override else fetch(part.source, part.ref, cache_dir)
        files.extend(_scan(root, part.dirs, split_value, default_group))
    return files


def _subsample(files: list, n: int | None, rng: random.Random) -> list:
    """At most n items per group, so every generator keeps the same share."""
    if n is None:
        return files
    by_group: dict[str, list] = {}
    for f in files:
        by_group.setdefault(f[1], []).append(f)
    out = []
    for group in sorted(by_group):
        items = by_group[group]
        out.extend(items if n >= len(items) else sorted(rng.sample(items, n)))
    return out


def load_dataset(
    name: str,
    split: str | None = None,
    transform: Callable | None = None,
    max_per_class: int | None = None,
    data_root: str | None = None,
    real_data_root: str | None = None,
    fake_data_root: str | None = None,
    cache_dir: str | None = None,
    seed: int = 0,
) -> ImageListDataset:
    """Fetch (if needed) and index a registered dataset.

    Args:
        name: key in DATASETS.
        split: split name; defaults to the spec's first split.
        transform: image transform (e.g. RIGIDDetector.transform).
        max_per_class: randomly keep at most this many images per group
            (real, and each fake generator).
        data_root: local folder to use for both real and fake parts (skips downloading).
        real_data_root / fake_data_root: local folder for just one side.
        cache_dir: where downloads are cached (e.g. a folder on an external drive).
        seed: seed for subsampling.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; available: {list(DATASETS)}")
    spec = DATASETS[name]
    split = split or next(iter(spec.splits))
    if split not in spec.splits:
        raise ValueError(f"{name} has splits {list(spec.splits)}, not {split!r}")
    value = spec.splits[split]

    real = _collect(spec.real, real_data_root or data_root, value, "real", cache_dir)
    fake = _collect(spec.fake, fake_data_root or data_root, value, "fake", cache_dir)
    if not real or not fake:
        raise RuntimeError(f"Found {len(real)} real and {len(fake)} fake images for "
                           f"{name} (split={split!r}); check the folder layout / paths.")

    rng = random.Random(seed)
    real = _subsample(real, max_per_class, rng)
    fake = _subsample(fake, max_per_class, rng)
    items = [(p, REAL, g) for p, g in real] + [(p, FAKE, g) for p, g in fake]
    return ImageListDataset(items, transform)
