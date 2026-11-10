"""Extract a random subset of images from a remote zip without downloading all of it.

Reads the zip's index over HTTP range requests, then fetches only the chosen files.
Works for any server that supports range requests, including public Google Drive files.

    # 1000 random ImageNet256 val images from the RIGID / dgm-eval Google Drive
    uv run python scripts/fetch_zip_subset.py --gdrive-id 1kbAvWrXw_GCE4ulMlx5zlfVOG_v6iMM7 \
        --prefix imagenet256/val/ --n 1000 --out "/Volumes/Kioxia SSD/datasets/rigid_imagenet/Imagenet"

Files keep their path inside the zip (minus --prefix). Re-running skips existing files.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def gdrive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="direct URL of the zip")
    src.add_argument("--gdrive-id", help="Google Drive file id of the zip")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000, help="number of images to fetch")
    parser.add_argument("--prefix", default="", help="only consider files under this path in the zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude-dir", type=Path, action="append", default=[],
                        help="skip zip members whose file name already exists anywhere under this folder")
    args = parser.parse_args()

    url = args.url or gdrive_url(args.gdrive_id)
    with RemoteZip(url) as z:
        members = [i.filename for i in z.infolist()
                   if not i.is_dir() and i.filename.startswith(args.prefix)
                   and PurePosixPath(i.filename).suffix.lower() in IMAGE_EXTS]
    if args.exclude_dir:
        seen = {p.name for d in args.exclude_dir for p in d.expanduser().rglob("*") if p.is_file()}
        before = len(members)
        members = [m for m in members if PurePosixPath(m).name not in seen]
        print(f"excluded {before - len(members)} already-used images")
    print(f"{len(members)} images in zip under {args.prefix or '/'}")
    chosen = sorted(random.Random(args.seed).sample(members, min(args.n, len(members))))

    out = args.out.expanduser()
    local = threading.local()

    def fetch(name: str) -> None:
        dest = out / PurePosixPath(name[len(args.prefix):])
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(6):  # servers (e.g. Google Drive) sometimes answer 503: back off and retry
            try:
                if not hasattr(local, "zip"):
                    local.zip = RemoteZip(url)  # one connection per thread
                tmp.write_bytes(local.zip.read(name))
                break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                if hasattr(local, "zip"):
                    del local.zip
                time.sleep(5 * 2 ** attempt)
        tmp.replace(dest)

    with ThreadPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(fetch, chosen), total=len(chosen), desc="Fetching"))
    print(f"Saved {len(chosen)} images to {out}")


if __name__ == "__main__":
    main()
