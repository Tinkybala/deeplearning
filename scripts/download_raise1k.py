"""Download the RAISE-1k TIFF images listed in the CSV from the RAISE website.

Get the CSV by filling in the form at https://loki.disi.unitn.it/RAISE/download.html
(RAISE-1k package). Then:

    uv run python scripts/download_raise1k.py --csv ~/Downloads/RAISE_1k.csv \
        --out "/Volumes/Kioxia SSD/datasets/raise1k"

Re-running skips files that are already complete, so an interrupted download can resume.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def download(url: str, dest: Path, retries: int = 3) -> str:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                expected = int(r.headers.get("Content-Length", 0))
                if dest.exists() and expected and dest.stat().st_size == expected:
                    return "skipped"
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
            if expected and tmp.stat().st_size != expected:
                raise OSError(f"size {tmp.stat().st_size} != {expected}")
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                return f"failed: {e}"
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True, help="RAISE_1k.csv from the RAISE site")
    parser.add_argument("--out", type=Path, required=True, help="folder to save TIFFs in")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.csv.expanduser(), newline="") as f:
        rows = list(csv.DictReader(f))
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    jobs = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for row in rows:
            url = row["TIFF"].strip()
            jobs[pool.submit(download, url, out / Path(url).name)] = row["File"]
        status = {"ok": 0, "skipped": 0}
        failed = []
        for fut in tqdm(as_completed(jobs), total=len(jobs), desc="RAISE-1k"):
            res = fut.result()
            if res in status:
                status[res] += 1
            else:
                failed.append((jobs[fut], res))

    print(f"Downloaded {status['ok']}, already present {status['skipped']}, failed {len(failed)}")
    for name, err in failed[:20]:
        print(f"  {name}: {err}")
    if failed:
        print("Re-run the same command to retry the failed files.")


if __name__ == "__main__":
    main()
