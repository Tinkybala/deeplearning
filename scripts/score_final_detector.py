"""Score images from any dataset with the exported final detector.

The final detector (models/final_detector_dinov2_l20_mlp.pt, made by
scripts/export_final_detector.py) is an MLP on the DINOv2 ViT-L/14 block-20 CLS token, trained
on all seven GenImage generators with clean + JPEG/resize-augmented images. On GenImage's
leave-one-generator-out protocol it reaches 81.3% mean balanced accuracy over the six test
conditions (88.7% clean). This script tests how it transfers to *other* datasets.

Give folders of real and of generated images (searched recursively). Each folder is reported
separately and in total:
    AUROC                    threshold-free ranking quality (real = positive)
    BA @ shipped threshold   balanced accuracy at the threshold calibrated on GenImage
    BA @ test GMM            label-free threshold from this test set's own scores (a
                             two-component Gaussian mixture on logit P(real))
    BA @ oracle              best possible threshold on these labels (upper bound)
plus real / generated recall at the shipped threshold. Per-image scores go to a CSV.

    python scripts/score_final_detector.py --name coco \\
        --real /data/coco_ai/real/coco \\
        --fake /data/coco_ai/fake/sd3 /data/coco_ai/fake/sd35 /data/coco_ai/fake/midjourney6 \\
        --max-per-folder 200

    python scripts/score_final_detector.py --name mine --images some/folder   # unlabelled: scores only

Outputs: results/external/<name>.json, .md and _scores.csv. The first run downloads DINOv2
ViT-L/14 (about 1.2 GB) through torch.hub. Needs torch, torchvision, scikit-learn, Pillow.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rigid import DEFAULT_MEAN, DEFAULT_STD, get_device, load_model  # noqa: E402
from rigid_probe import make_mlp  # noqa: E402

MODEL = ROOT / "models" / "final_detector_dinov2_l20_mlp.pt"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)])


class Detector:
    def __init__(self, path: Path = MODEL, device: str | None = None):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.meta, self.threshold = ck["meta"], ck["threshold_p_real"]
        self.layer = int(self.meta["layer"])
        self.mean, self.scale = ck["scaler_mean"], ck["scaler_scale"]
        self.net = make_mlp(ck["in_dim"], ck["hidden"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.device = get_device(device)
        self.backbone = load_model(self.meta["backbone"], device=str(self.device))

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        (_, cls), = self.backbone.get_intermediate_layers(x.to(self.device), n=[self.layer - 1],
                                                         return_class_token=True, norm=True)
        return cls.float().cpu()

    @torch.no_grad()
    def p_real(self, feats: torch.Tensor) -> np.ndarray:
        z = (feats.half().float() - self.mean) / self.scale      # float16 round trip as in training
        return torch.sigmoid(self.net(z)).squeeze(1).numpy()


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return TRANSFORM(im.convert("RGB")), i
        except Exception as err:                                 # unreadable file: report and skip
            print(f"skipping {self.paths[i]}: {err!r}", file=sys.stderr)
            return torch.zeros(3, 224, 224), -1


def list_images(folder: Path, limit: int | None, seed: int) -> list[Path]:
    files = sorted(p for p in folder.expanduser().rglob("*") if p.suffix.lower() in EXTS and p.is_file()
                   and not any(part.startswith(".") for part in p.parts))
    if limit and len(files) > limit:
        files = sorted(random.Random(seed).sample(files, limit))
    return files


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def gmm_threshold(s: np.ndarray) -> float:
    gm = GaussianMixture(2, random_state=0, reg_covar=1e-4).fit(s.reshape(-1, 1))
    mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
    lo, hi = np.argsort(mu)
    x = np.linspace(mu[lo], mu[hi], 2001)
    lp = lambda m, v: -np.log(v) - 0.5 * ((x - m) / v) ** 2  # noqa: E731
    return float(x[np.argmin(np.abs(lp(mu[lo], sd[lo]) - lp(mu[hi], sd[hi])))])


def ba(y, pred_real):
    return 0.5 * (pred_real[y == 1].mean() + (~pred_real[y == 0]).mean())


def oracle_ba(y, s):
    order = np.argsort(s)
    ys, ss = y[order] == 1, s[order]
    uniq, first = np.unique(ss, return_index=True)
    real_below = np.concatenate([[0], np.cumsum(ys)])[first]
    vals = 0.5 * ((ys.sum() - real_below) / ys.sum() + (first - real_below) / (~ys).sum())
    return float(vals.max())


def summary(y, p, thr):
    s = logit(p)
    out = {"n_real": int((y == 1).sum()), "n_generated": int((y == 0).sum())}
    if out["n_real"] and out["n_generated"]:
        out |= {"auroc": float(roc_auc_score(y, p)), "ba_shipped": float(ba(y, p >= thr)),
                "ba_test_gmm": float(ba(y, s >= gmm_threshold(s))), "ba_oracle": oracle_ba(y, p)}
    if out["n_real"]:
        out["real_recall"] = float((p[y == 1] >= thr).mean())
    if out["n_generated"]:
        out["generated_recall"] = float((p[y == 0] < thr).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="name of this test set (output file names)")
    parser.add_argument("--real", nargs="*", type=Path, default=[], help="folders of real images")
    parser.add_argument("--fake", nargs="*", type=Path, default=[], help="folders of generated images")
    parser.add_argument("--images", nargs="*", type=Path, default=[], help="unlabelled folders (scores only)")
    parser.add_argument("--max-per-folder", type=int, default=None, help="random sample per folder (seeded)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external")
    args = parser.parse_args()
    if not (args.real or args.fake or args.images):
        parser.error("give --real/--fake folders, or --images")

    groups = [(f, 1) for f in args.real] + [(f, 0) for f in args.fake] + [(f, -1) for f in args.images]
    items = []
    for folder, label in groups:
        files = list_images(folder, args.max_per_folder, args.seed)
        if not files:
            parser.error(f"no images found under {folder}")
        print(f"{len(files):6d} images  {'real' if label == 1 else 'generated' if label == 0 else 'unlabelled'}  {folder}")
        items += [(p, label, folder.name) for p in files]

    det = Detector(args.model, args.device)
    loader = DataLoader(Images([p for p, _, _ in items]), batch_size=args.batch_size, num_workers=args.workers)
    p = np.full(len(items), np.nan)
    for x, idx in loader:
        keep = idx >= 0
        if keep.any():
            p[idx[keep].numpy()] = det.p_real(det.features(x[keep]))
    ok = ~np.isnan(p)
    y = np.asarray([lab for _, lab, _ in items])
    src = np.asarray([s for _, _, s in items])
    thr = det.threshold

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / f"{args.name}_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", "p_real", "prediction"])
        for (path, lab, s), pr in zip(items, p):
            w.writerow([str(path), s, {1: "real", 0: "generated", -1: ""}[lab],
                        "" if np.isnan(pr) else f"{pr:.6f}", "" if np.isnan(pr) else ("real" if pr >= thr else "generated")])

    report = {"model": str(args.model.name), "threshold_p_real": thr, "meta": det.meta, "skipped": int((~ok).sum())}
    lab = ok & (y >= 0)
    if lab.any():
        report["overall"] = summary(y[lab], p[lab], thr)
        reals = lab & (y == 1)
        report["per_generated_source"] = {s: summary(np.concatenate([y[reals], y[lab & (src == s) & (y == 0)]]),
                                                     np.concatenate([p[reals], p[lab & (src == s) & (y == 0)]]), thr)
                                          for s in dict.fromkeys(src[lab & (y == 0)])}
        report["per_real_source"] = {s: summary(y[lab & (src == s) & (y == 1)], p[lab & (src == s) & (y == 1)], thr)
                                     for s in dict.fromkeys(src[reals])}
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(report, indent=1))

    def fmt(d):
        cells = [("" if k not in d else f"{d[k]:.3f}" if k == "auroc" else f"{d[k]:.1%}")
                 for k in ("auroc", "ba_shipped", "ba_test_gmm", "ba_oracle", "real_recall", "generated_recall")]
        return " | ".join(cells)
    lines = [f"# Final detector on {args.name}", f"Shipped threshold: P(real) >= {thr:.4f}. Skipped files: {report['skipped']}."]
    if "overall" in report:
        head = "| Set | AUROC | BA @ shipped | BA @ test GMM | BA @ oracle | Real recall | Generated recall |\n|---|---:|---:|---:|---:|---:|---:|"
        rows = [f"| all ({report['overall']['n_real']} real, {report['overall']['n_generated']} generated) | {fmt(report['overall'])} |"]
        rows += [f"| {s} vs all real | {fmt(d)} |" for s, d in report["per_generated_source"].items()]
        rows += [f"| real: {s} | {fmt(d)} |" for s, d in report["per_real_source"].items()]
        lines.append(head + "\n" + "\n".join(rows))
    (args.out_dir / f"{args.name}.md").write_text("\n\n".join(lines) + "\n")
    print("\n\n".join(lines))
    print(f"\nSaved -> {args.out_dir / args.name}.{{json,md}} and {args.name}_scores.csv")


if __name__ == "__main__":
    main()
