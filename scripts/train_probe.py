"""Train real/fake probes on cached DINOv2 embeddings and compare feature sets.

Feature sets (all from scripts/extract_embeddings.py):
    rigid        - plain RIGID cosine score, no training (higher = real)
    cos          - the cosine score through a learned 1-D classifier
    clean        - clean embedding only (UnivFD-style linear probe, on DINOv2)
    diff         - noisy - clean only
    clean+cos    - clean embedding + cosine score
    clean+diff   - clean embedding + (noisy - clean)
    clean+noisy  - both embeddings concatenated

Protocols:
    within  - train and test on the same dataset (5-fold CV; CIFAKE uses its train/test split).
              Synthbuster folds keep each RAISE scene and its fakes together.
    logo    - leave one generator out: train on the other generators, test on the held-out one,
              with real images split so no real image (or scene) is in both train and test.
    cross   - train on all of one dataset, test on all of another.

Usage:
    uv run python scripts/train_probe.py [--emb-dir results/embeddings] [--models logreg mlp]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["rigid", "cos", "clean", "diff", "clean+cos", "clean+diff", "clean+noisy"]


# ---------------------------------------------------------------- data
def load(emb_dir: Path) -> dict:
    parts = [np.load(f) for f in sorted(emb_dir.glob("*.npz"))]
    if not parts:
        sys.exit(f"No .npz files in {emb_dir}; run scripts/extract_embeddings.py first")
    d = {k: np.concatenate([p[k] for p in parts]) for k in ["clean", "noisy", "cos", "label", "group",
                                                            "source", "dataset", "path"]}
    d["clean"] = d["clean"].astype(np.float32)
    d["noisy"] = d["noisy"].astype(np.float32)
    d["split"] = np.array(["test" if "/test/" in p else "train" for p in d["path"]])  # CIFAKE only
    return d


def features(d: dict, name: str, idx: np.ndarray) -> np.ndarray:
    c, n, cos = d["clean"][idx], d["noisy"][idx], d["cos"][idx, None]
    return {"cos": cos, "clean": c, "diff": n - c, "clean+cos": np.hstack([c, cos]),
            "clean+diff": np.hstack([c, n - c]), "clean+noisy": np.hstack([c, n])}[name]


# ---------------------------------------------------------------- models
def fit_predict_logreg(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def fit_predict_mlp(Xtr, ytr, Xte, seed, epochs=300, hidden=256):
    """Small MLP, full-batch Adam, class-balanced loss, early stopping on a 10% split."""
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    Xte_t = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    perm = torch.randperm(len(y))
    nval = max(1, len(y) // 10)
    va, tr = perm[:nval], perm[nval:]
    pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr_t.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss_fn(net(Xtr_t[tr]).squeeze(1), y[tr]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xtr_t[va]).squeeze(1), y[va]).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(Xte_t).squeeze(1).numpy()


MODELS = {"logreg": fit_predict_logreg, "mlp": fit_predict_mlp}


def run(d, feat, model, tr, te, seed=0) -> np.ndarray:
    """Scores for test indices (higher = real)."""
    if feat == "rigid":
        return d["cos"][te]
    return MODELS[model](features(d, feat, tr), d["label"][tr], features(d, feat, te), seed)


def auc(y, s) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


# ---------------------------------------------------------------- protocols
def within(d, feat, model) -> list[dict]:
    rows = []
    for ds in np.unique(d["dataset"]):
        idx = np.where(d["dataset"] == ds)[0]
        if ds == "cifake":
            tr, te = idx[d["split"][idx] == "train"], idx[d["split"][idx] == "test"]
            folds = [(tr, te)]
        elif ds == "synthbuster":
            folds = [(idx[a], idx[b]) for a, b in GroupKFold(5).split(idx, groups=d["group"][idx])]
        else:
            folds = [(idx[a], idx[b]) for a, b in
                     StratifiedKFold(5, shuffle=True, random_state=0).split(idx, d["label"][idx])]
        oof_idx, oof = [], []
        for k, (tr, te) in enumerate(folds):
            oof_idx.append(te)
            oof.append(run(d, feat, model, tr, te, seed=k))
        te, s = np.concatenate(oof_idx), np.concatenate(oof)
        y, src = d["label"][te], d["source"][te]
        rows.append({"protocol": "within", "train": ds, "test": ds, "test_source": "all", "auroc": auc(y, s)})
        for g in np.unique(src[y == 0]):     # per-generator: that generator vs all real
            m = (y == 1) | (src == g)
            rows.append({"protocol": "within", "train": ds, "test": ds, "test_source": g, "auroc": auc(y[m], s[m])})
    return rows


def logo(d, feat, model) -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    for ds in ["synthbuster", "imagenet"]:
        idx = np.where(d["dataset"] == ds)[0]
        if len(idx) == 0:
            continue
        # Split scenes / real images into halves A and B; train on A, test on B, then swap.
        groups = np.unique(d["group"][idx])
        half = set(rng.permutation(groups)[: len(groups) // 2])
        in_a = np.array([g in half for g in d["group"][idx]])
        gens = np.unique(d["source"][idx][d["label"][idx] == 0])
        for g in gens:
            aucs = []
            for a_side in (True, False):
                tr_m = (in_a == a_side) & (d["source"][idx] != g)
                te_m = (in_a != a_side) & ((d["label"][idx] == 1) | (d["source"][idx] == g))
                tr, te = idx[tr_m], idx[te_m]
                aucs.append(auc(d["label"][te], run(d, feat, model, tr, te)))
            rows.append({"protocol": "logo", "train": ds, "test": ds, "test_source": g, "auroc": np.mean(aucs)})
    return rows


def cross(d, feat, model) -> list[dict]:
    rows = []
    for a, b in itertools.permutations(np.unique(d["dataset"]), 2):
        tr, te = np.where(d["dataset"] == a)[0], np.where(d["dataset"] == b)[0]
        rows.append({"protocol": "cross", "train": a, "test": b, "test_source": "all",
                     "auroc": auc(d["label"][te], run(d, feat, model, tr, te))})
    return rows


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emb-dir", type=Path, default=ROOT / "results/embeddings")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/probe")
    parser.add_argument("--models", nargs="+", default=["logreg", "mlp"], choices=list(MODELS))
    parser.add_argument("--features", nargs="+", default=FEATURES, choices=FEATURES)
    parser.add_argument("--protocols", nargs="+", default=["within", "logo", "cross"])
    args = parser.parse_args()

    d = load(args.emb_dir)
    print("Loaded:", pd.Series(d["dataset"]).value_counts().to_dict(),
          "| real/fake:", int(d["label"].sum()), "/", int((d["label"] == 0).sum()))
    proto_fns = {"within": within, "logo": logo, "cross": cross}
    rows = []
    for model in args.models:
        for feat in args.features:
            if feat == "rigid" and model != args.models[0]:
                continue                       # no training, only needs one row
            for p in args.protocols:
                for r in proto_fns[p](d, feat, model):
                    rows.append({**r, "model": "none" if feat == "rigid" else model, "features": feat})
            print(f"  done: {model:<6} {feat}")
    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "probe_results.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50)
    col = lambda m, f: "RIGID (no training)" if f == "rigid" else f"{m}:{f}"
    df["method"] = [col(m, f) for m, f in zip(df.model, df.features)]
    order = [c for c in ["RIGID (no training)"] + [f"{m}:{f}" for m in args.models for f in args.features if f != "rigid"]
             if c in set(df.method)]

    print("\n=== WITHIN-dataset AUROC (overall) ===")
    w = df[(df.protocol == "within") & (df.test_source == "all")]
    print(w.pivot(index="method", columns="test", values="auroc").reindex(order).round(3).to_string())
    print("\n=== WITHIN Synthbuster, per generator ===")
    ws = df[(df.protocol == "within") & (df.train == "synthbuster") & (df.test_source != "all")]
    if len(ws):
        print(ws.pivot(index="method", columns="test_source", values="auroc").reindex(order).round(3).to_string())
    print("\n=== LEAVE-ONE-GENERATOR-OUT AUROC (held-out generator vs unseen real) ===")
    lg = df[df.protocol == "logo"]
    if len(lg):
        t = lg.pivot_table(index="method", columns=["train", "test_source"], values="auroc").reindex(order)
        t[("", "mean SB")] = lg[lg.train == "synthbuster"].groupby("method").auroc.mean().reindex(order)
        print(t.round(3).to_string())
    print("\n=== CROSS-dataset AUROC (train -> test) ===")
    cr = df[df.protocol == "cross"].assign(pair=lambda x: x.train + " -> " + x.test)
    if len(cr):
        print(cr.pivot(index="method", columns="pair", values="auroc").reindex(order).round(3).to_string())
    print(f"\nSaved {args.out_dir / 'probe_results.csv'}")


if __name__ == "__main__":
    main()
