"""Where datasets come from.

Each fetcher takes a source-specific reference and returns a local directory.
To support a new source (Hugging Face, a zip URL, S3, ...), add a function here
and register it in SOURCES.
"""

from __future__ import annotations

import os
from pathlib import Path


def fetch_kaggle(handle: str, cache_dir: str | None = None) -> Path:
    """Download a Kaggle dataset and return its folder (cached; later calls are instant).

    `cache_dir` overrides the default cache (~/.cache/kagglehub), e.g. to keep large
    datasets on an external drive. Public datasets need no credentials; for private ones
    set KAGGLE_USERNAME / KAGGLE_KEY or put kaggle.json in ~/.kaggle.
    """
    if cache_dir:
        os.environ["KAGGLEHUB_CACHE"] = str(Path(cache_dir).expanduser())
    import kagglehub

    return Path(kagglehub.dataset_download(handle))


def fetch_local(path: str, cache_dir: str | None = None) -> Path:
    """A dataset already on disk."""
    if not path:
        raise ValueError("This dataset part is a local folder; set its path in the config "
                         "(DATA_ROOT, REAL_DATA_ROOT or FAKE_DATA_ROOT).")
    p = Path(path).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {p}")
    return p


SOURCES = {
    "kaggle": fetch_kaggle,
    "local": fetch_local,
}


def fetch(source: str, ref: str, cache_dir: str | None = None) -> Path:
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; available: {list(SOURCES)}")
    return SOURCES[source](ref, cache_dir)
