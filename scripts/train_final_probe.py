"""Train the final RIGID probe on cached embeddings and save it for rigid_probe.ProbeDetector.

    uv run python scripts/train_final_probe.py                       # clean+diff, all datasets
    uv run python scripts/train_final_probe.py --datasets synthbuster --out models/probe_sb.joblib

Uses embeddings from scripts/extract_embeddings.py; the noise strength, backbone and
image size are read from that config and stored with the probe so inference matches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rigid_probe import build_features, make_mlp  # noqa: E402


def train_mlp(Xs: np.ndarray, y: np.ndarray, hidden: int, seed: int = 0, epochs: int = 300):
    """Same recipe as scripts/train_probe.py: full-batch AdamW, class-balanced loss,
    early stopping on a 10% validation split. Returns the trained (eval-mode) network."""
    torch.manual_seed(seed)
    X_t, y_t = torch.tensor(Xs, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    perm = torch.randperm(len(y_t))
    nval = max(1, len(y_t) // 10)
    va, tr = perm[:nval], perm[nval:]
    pos_w = (y_t[tr] == 0).sum() / max((y_t[tr] == 1).sum(), 1)
    net = make_mlp(X_t.shape[1], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss_fn(net(X_t[tr]).squeeze(1), y_t[tr]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(X_t[va]).squeeze(1), y_t[va]).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(best_state)
    return net.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emb-config", type=Path, default=ROOT / "configs/embeddings.yaml")
    parser.add_argument("--features", default="clean+diff")
    parser.add_argument("--datasets", nargs="+", default=None, help="default: every dataset with embeddings")
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--C", type=float, default=0.1, help="logreg: inverse L2 regularization strength")
    parser.add_argument("--hidden", type=int, default=256, help="mlp: hidden units")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: models/probe_<features>[_mlp].joblib")
    args = parser.parse_args()

    cfg = {"LAMB": 0.05, "MODEL_NAME": "dinov2_vitl14", "IMAGE_SIZE": 224, "SEED": 100,
           "OUTPUT_DIR": "results/embeddings", **yaml.safe_load(args.emb_config.read_text())}
    files = sorted((ROOT / cfg["OUTPUT_DIR"]).glob("*.npz"))
    parts = [np.load(f) for f in files]
    parts = [p for p in parts if args.datasets is None or str(p["dataset"][0]) in args.datasets]
    if not parts:
        sys.exit("No matching embeddings found")
    clean = np.concatenate([p["clean"] for p in parts]).astype(np.float32)
    noisy = np.concatenate([p["noisy"] for p in parts]).astype(np.float32)
    y = np.concatenate([p["label"] for p in parts])
    datasets = sorted({str(p["dataset"][0]) for p in parts})
    sources = sorted({str(p["source"][0]) for p in parts})

    X = build_features(clean, noisy, args.features)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    meta = {"model": args.model, "features": args.features, "lamb": cfg["LAMB"],
            "model_name": cfg["MODEL_NAME"], "image_size": cfg["IMAGE_SIZE"], "seed": cfg["SEED"],
            "datasets": datasets, "sources": sources, "n_real": int(y.sum()), "n_fake": int((y == 0).sum())}
    if args.model == "mlp":
        net = train_mlp(Xs, y, args.hidden)
        with torch.no_grad():
            train_scores = net(torch.tensor(Xs, dtype=torch.float32)).squeeze(1).numpy()
        meta.update(in_dim=Xs.shape[1], hidden=args.hidden)
        bundle = {"scaler": scaler, "state_dict": net.state_dict(), "meta": meta}
    else:
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=5000).fit(Xs, y)
        train_scores = clf.decision_function(Xs)
        meta.update(C=args.C)
        bundle = {"scaler": scaler, "clf": clf, "meta": meta}
    train_auc = roc_auc_score(y, train_scores)
    meta["train_auroc"] = float(train_auc)

    out = args.out or ROOT / f"models/probe_{args.features.replace('+', '_')}{'_mlp' if args.model == 'mlp' else ''}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"Trained {args.model} on {datasets}: {meta['n_real']} real + {meta['n_fake']} fake "
          f"({len(sources) - 2} generators); features={args.features}")
    print(f"Training AUROC {train_auc:.4f} (optimistic; see results/probe for held-out numbers)")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
