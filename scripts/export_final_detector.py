"""Train the final detector on all seven GenImage generators and save it for external tests.

Final detector (final_detector_experiments.py): MLP probe (256 hidden units, the main runs'
recipe) on the standardised DINOv2 ViT-L/14 block-20 CLS token, fit on clean + augmented
images (augmentation "both": JPEG 30-95 and/or resize to 32/64/128 px, p = 0.5 each).
Leave-one-generator-out: 81.3% mean balanced accuracy over the six standard conditions.

The exported model is trained on every generator, so it is meant for *other* datasets
(scripts/score_final_detector.py), not for GenImage test numbers.

  fit          all 7 generators' train/ai (14,000) + the seed-42 80% fit split of BigGAN's and
               VQDM's train/nature (3,200 distinct real photos), each clean and augmented
  threshold    balanced-accuracy-optimal P(real) on the clean calibration set: the other 20% of
               those real photos (800) + every generator's val/ai (3,500)

Output: models/final_detector_dinov2_l20_mlp.pt (about 1 MB; tensors and plain values only,
loadable with torch.load(weights_only=True)). Uses only cached features.

    python scripts/export_final_detector.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backbone_comparison as bc  # noqa: E402
import cached_probe_experiments as cpe  # noqa: E402
from train_final_probe import train_mlp  # noqa: E402

resnet = bc.dh.resnet
OUT = ROOT / "models" / "final_detector_dinov2_l20_mlp.pt"
KEYS = ["dino_cls20"]
AUG = "clean__aug-both"


def main() -> None:
    args = type("A", (), {})()
    cpe.set_defaults(args)
    G = resnet.GENERATORS
    fit_rec, cal_rec = [], []
    for src in G[:2]:                                   # BigGAN and VQDM real photos (disjoint sets)
        real = resnet.files_for(args.data_root, src, "train", "nature")
        random.Random(args.seed).shuffle(real)
        k = max(1, len(real) // 5)
        fit_rec += [(p, 1, "real") for p in real[k:]]
        cal_rec += [(p, 1, "real") for p in real[:k]]
    fit_rec += [(p, 0, g) for g in G for p in resnet.files_for(args.data_root, g, "train", "ai")]
    cal_rec += [(p, 0, g) for g in G for p in resnet.files_for(args.data_root, g, "val", "ai")]
    cal_rec = bc.fa.shared_val(args, cal_rec)           # val/ai == test/<g>/test/ai, whose features are cached

    fstore = bc.FeatureStore()
    y = np.asarray([r[1] for r in fit_rec])
    X = np.vstack([fstore.gather(fit_rec, "clean", KEYS), fstore.gather(fit_rec, AUG, KEYS)])
    y2 = np.concatenate([y, y])
    scaler = StandardScaler().fit(X)
    net = train_mlp(scaler.transform(X), y2, args.hidden, seed=args.seed)
    Xc = fstore.gather(cal_rec, "clean", KEYS)
    yc = np.asarray([r[1] for r in cal_rec])
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(scaler.transform(Xc), dtype=torch.float32))).squeeze(1).numpy()
    thr, ba = cpe.best_threshold(yc, p)
    OUT.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(), "in_dim": int(X.shape[1]), "hidden": int(args.hidden),
        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "threshold_p_real": float(thr),
        "meta": {"backbone": "dinov2_vitl14", "layer": 20, "token": "CLS, layer-normed (get_intermediate_layers norm=True)",
                 "preprocess": "Resize(256) + CenterCrop(224) + ImageNet normalisation",
                 "output": "sigmoid = P(real); P(real) >= threshold -> real",
                 "fit": f"{int((y == 1).sum())} real + {int((y == 0).sum())} generated GenImage training images "
                        "(7 generators), each clean and augmented (JPEG 30-95 / resize 32-128 px, p = 0.5 each)",
                 "calibration": f"{int((yc == 1).sum())} real + {int((yc == 0).sum())} generated, clean; "
                                f"balanced accuracy at the threshold {ba:.4f}",
                 "seed": int(args.seed)},
    }, OUT)
    print(f"Saved {OUT.relative_to(ROOT)}: threshold P(real) >= {thr:.4f} (calibration BA {ba:.3f}), "
          f"fit on {len(y2)} feature rows")


if __name__ == "__main__":
    main()
