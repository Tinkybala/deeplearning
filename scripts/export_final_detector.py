"""Train a final detector on all seven GenImage generators and save it for external tests.

--detector dinov2 (default): the final detector (final_detector_experiments.py), an MLP probe
(256 hidden units, the main runs' recipe) on the standardised DINOv2 ViT-L/14 block-20 CLS
token, fit on clean + augmented images (augmentation "both": JPEG 30-95 and/or resize to
32/64/128 px, p = 0.5 each). Leave-one-generator-out: 81.3% mean balanced accuracy over the
six standard conditions.

--detector fusion: the nested pick of clip_layer_experiments.py --backbone dinov3, in all 7
folds. The same DINOv2 head, plus the same kind of MLP on DINOv3 ViT-L/16's block-24 CLS token
(timm's vit_large_patch16_dinov3.lvd1689m, through its final norm). Each head's logit is
standardised on the calibration set and the two are averaged. Leave-one-generator-out: 84.4%
(+3.1 pp over the DINOv2 detector, better in every condition).

The exported models are trained on every generator, so they are meant for *other* datasets
(scripts/score_final_detector.py), not for GenImage test numbers.

  fit          all 7 generators' train/ai (14,000) + the seed-42 80% fit split of BigGAN's and
               VQDM's train/nature (3,200 distinct real photos), each clean and augmented
  calibration  the other 20% of those real photos (800) + every generator's val/ai (3,500),
               clean: the fusion's standardisation and both detectors' balanced-accuracy-optimal
               threshold

Output (tensors and plain values only, loadable with torch.load(weights_only=True)):
  models/final_detector_dinov2_l20_mlp.pt           about 1 MB; threshold on P(real)
  models/fusion_detector_dinov2_l20_dinov3_l24.pt   about 2 MB; threshold on the fused score
Uses only cached features (the fusion needs clip_layer_features.py --backbone dinov3).

    python scripts/export_final_detector.py [--detector fusion]
"""

from __future__ import annotations

import argparse
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
OUT = {"dinov2": ROOT / "models" / "final_detector_dinov2_l20_mlp.pt",
       "fusion": ROOT / "models" / "fusion_detector_dinov2_l20_dinov3_l24.pt"}
AUG = "clean__aug-both"
PREPROCESS = "Resize(256) + CenterCrop(224) + ImageNet normalisation"
# backbone -> (cache suffix, feature key, layer, token description)
HEADS = {"dinov2_vitl14": ("", "dino_cls20", 20, "CLS, layer-normed (get_intermediate_layers norm=True)"),
         "dinov3_vitl16": ("__dinov3", "dinov3_cls24", 24,
                           "CLS after the block, through the final norm (timm vit_large_patch16_dinov3.lvd1689m)")}


def records(args):
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
    return fit_rec, cal_rec


def fit_head(args, fstore, fit_rec, cal_rec, backbone):
    """MLP on one backbone's feature, clean + augmented; returns its saved fields and P(real) on calibration."""
    suffix, key, layer, token = HEADS[backbone]
    y = np.asarray([r[1] for r in fit_rec])
    X = np.vstack([fstore.gather(fit_rec, "clean" + suffix, [key]), fstore.gather(fit_rec, AUG + suffix, [key])])
    scaler = StandardScaler().fit(X)
    net = train_mlp(scaler.transform(X), np.concatenate([y, y]), args.hidden, seed=args.seed)
    with torch.no_grad():
        Xc = scaler.transform(fstore.gather(cal_rec, "clean" + suffix, [key]))
        p = torch.sigmoid(net(torch.tensor(Xc, dtype=torch.float32))).squeeze(1).numpy()
    head = {"backbone": backbone, "layer": layer, "token": token,
            "state_dict": net.state_dict(), "in_dim": int(X.shape[1]), "hidden": int(args.hidden),
            "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
            "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32)}
    return head, p, int((y == 1).sum()), int((y == 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=list(OUT), default="dinov2")
    cli = parser.parse_args()
    args = type("A", (), {})()
    cpe.set_defaults(args)
    fit_rec, cal_rec = records(args)
    yc = np.asarray([r[1] for r in cal_rec])
    fstore = bc.FeatureStore()
    out = OUT[cli.detector]
    out.parent.mkdir(exist_ok=True)

    head, p, n_real, n_gen = fit_head(args, fstore, fit_rec, cal_rec, "dinov2_vitl14")
    fit_note = (f"{n_real} real + {n_gen} generated GenImage training images (7 generators), each clean and "
                "augmented (JPEG 30-95 / resize 32-128 px, p = 0.5 each)")
    if cli.detector == "dinov2":
        thr, ba = cpe.best_threshold(yc, p)
        torch.save({k: head[k] for k in ("state_dict", "in_dim", "hidden", "scaler_mean", "scaler_scale")} | {
            "threshold_p_real": float(thr),
            "meta": {"backbone": "dinov2_vitl14", "layer": 20, "token": head["token"], "preprocess": PREPROCESS,
                     "output": "sigmoid = P(real); P(real) >= threshold -> real", "fit": fit_note,
                     "calibration": f"{int((yc == 1).sum())} real + {int((yc == 0).sum())} generated, clean; "
                                    f"balanced accuracy at the threshold {ba:.4f}",
                     "seed": int(args.seed)},
        }, out)
        print(f"Saved {out.relative_to(ROOT)}: threshold P(real) >= {thr:.4f} (calibration BA {ba:.3f})")
        return

    heads, z = [], []
    for backbone, (h, ph) in {"dinov2_vitl14": (head, p),
                              "dinov3_vitl16": fit_head(args, fstore, fit_rec, cal_rec, "dinov3_vitl16")[:2]}.items():
        s = cpe.logit(ph)
        h |= {"z_mean": float(s.mean()), "z_std": float(s.std())}
        heads.append(h)
        z.append((s - h["z_mean"]) / h["z_std"])
        print(f"{backbone}: calibration BA at its own best threshold {cpe.best_threshold(yc, ph)[1]:.3f}")
    fused = np.mean(z, axis=0)
    thr, ba = cpe.best_threshold(yc, fused)
    torch.save({
        "kind": "fusion", "heads": heads, "threshold": float(thr),
        "meta": {"backbone": "dinov2_vitl14 block 20 + dinov3_vitl16 block 24", "preprocess": PREPROCESS,
                 "output": "per head: s = logit(sigmoid(MLP)), z = (s - z_mean) / z_std; fused = mean of the heads' z; "
                           "fused >= threshold -> real",
                 "fit": fit_note,
                 "calibration": f"{int((yc == 1).sum())} real + {int((yc == 0).sum())} generated, clean; z statistics "
                                f"and threshold; balanced accuracy at the threshold {ba:.4f}",
                 "seed": int(args.seed)},
    }, out)
    print(f"Saved {out.relative_to(ROOT)}: threshold fused >= {thr:.4f} (calibration BA {ba:.3f})")


if __name__ == "__main__":
    main()
