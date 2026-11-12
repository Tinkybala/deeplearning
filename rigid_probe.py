"""RIGID + trained probe: classify an image as real / AI-generated from DINOv2 embeddings.

Plain RIGID scores an image by the cosine similarity between DINOv2 embeddings of the
image and a noise-perturbed copy. Our experiments showed the *direction* the embedding
moves under noise carries much more information, so this detector feeds the clean
embedding and the change (noisy - clean) to a logistic-regression probe trained on
labelled real and fake images (see scripts/train_final_probe.py).

Usage:
    from rigid_probe import ProbeDetector
    det = ProbeDetector("model/DINOv2/Frozen DINO model/probes/probe_clean_diff.joblib")
    print(det.predict("some_image.png"))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rigid import RIGIDDetector


def build_features(clean: np.ndarray, noisy: np.ndarray, kind: str) -> np.ndarray:
    """Feature vector(s) from clean and noisy embeddings, shape (N, D)."""
    clean, noisy = np.asarray(clean, np.float32), np.asarray(noisy, np.float32)
    num = (clean * noisy).sum(1)
    cos = (num / (np.linalg.norm(clean, axis=1) * np.linalg.norm(noisy, axis=1)))[:, None]
    feats = {
        "clean": clean,
        "diff": noisy - clean,
        "cos": cos,
        "clean+cos": np.hstack([clean, cos]),
        "clean+diff": np.hstack([clean, noisy - clean]),
        "clean+noisy": np.hstack([clean, noisy]),
    }
    if kind not in feats:
        raise ValueError(f"Unknown feature set {kind!r}; choose from {list(feats)}")
    return feats[kind]


def make_mlp(in_dim: int, hidden: int = 256, dropout: float = 0.3) -> torch.nn.Module:
    """Small MLP probe: Linear -> GELU -> Dropout -> Linear (one logit, positive = real)."""
    return torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.GELU(),
                               torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))


@dataclass
class ProbeResult:
    path: str | None
    p_fake: float               # probe probability that the image is AI-generated
    p_fake_std: float           # spread across noise draws
    threshold: float
    label: str                  # "ai-generated" or "real"
    rigid_similarity: float     # plain RIGID score, for comparison (higher = more "real")

    def __str__(self) -> str:
        return (f"P(AI-generated) = {self.p_fake:.3f} (± {self.p_fake_std:.3f})  ->  {self.label.upper()} "
                f"(threshold {self.threshold:.2f})  |  plain RIGID similarity = {self.rigid_similarity:.4f}")


class ProbeDetector:
    """Loads a probe saved by scripts/train_final_probe.py, plus the matching DINOv2 setup."""

    def __init__(self, probe_path: str | Path, device: str | None = None, threshold: float = 0.5):
        probe_path = Path(probe_path).expanduser()
        if not probe_path.is_absolute() and not probe_path.exists():
            probe_path = Path(__file__).resolve().parent / probe_path   # relative to the project
        bundle = joblib.load(probe_path)
        self.meta: dict = bundle["meta"]
        self.scaler = bundle["scaler"]
        self.kind: str = self.meta.get("model", "logreg")      # "logreg" or "mlp"
        if self.kind == "mlp":
            self.clf = make_mlp(self.meta["in_dim"], self.meta["hidden"])
            self.clf.load_state_dict(bundle["state_dict"])
            self.clf.eval()
        else:
            self.clf = bundle["clf"]
        self.threshold = threshold
        # Same backbone, preprocessing and noise strength the probe was trained with.
        self.rigid = RIGIDDetector(lamb=self.meta["lamb"], model_name=self.meta["model_name"],
                                   device=device, image_size=self.meta["image_size"], seed=self.meta["seed"])

    @property
    def device(self) -> torch.device:
        return self.rigid.device

    @torch.no_grad()
    def embed(self, image, n_noise: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Clean embedding (n_noise, D) repeated, and noisy embeddings (n_noise, D)."""
        x = self.rigid.preprocess(image)
        gen = torch.Generator().manual_seed(self.rigid.seed) if self.rigid.seed is not None else None
        noise = torch.randn((n_noise, *x.shape[1:]), generator=gen).to(self.device)
        clean = self.rigid.model(x).float()
        noisy = self.rigid.model(x + noise * self.rigid.lamb).float()
        return clean.expand(n_noise, -1).cpu().numpy(), noisy.cpu().numpy()

    def predict(self, image: str | Path | Image.Image, n_noise: int = 1,
                threshold: float | None = None) -> ProbeResult:
        """Probability the image is AI-generated, averaged over `n_noise` noise draws."""
        threshold = self.threshold if threshold is None else threshold
        clean, noisy = self.embed(image, n_noise)
        X = self.scaler.transform(build_features(clean, noisy, self.meta["features"]))
        if self.kind == "mlp":
            with torch.no_grad():
                p_real = torch.sigmoid(self.clf(torch.tensor(X, dtype=torch.float32))).squeeze(1).numpy()
        else:
            p_real = self.clf.predict_proba(X)[:, list(self.clf.classes_).index(1)]
        p_fake = 1.0 - p_real
        rigid_sim = F.cosine_similarity(torch.from_numpy(clean), torch.from_numpy(noisy), dim=-1).mean().item()
        mean = float(p_fake.mean())
        return ProbeResult(
            path=None if isinstance(image, Image.Image) else str(image),
            p_fake=mean, p_fake_std=float(p_fake.std()) if n_noise > 1 else 0.0,
            threshold=threshold, label="ai-generated" if mean >= threshold else "real",
            rigid_similarity=rigid_sim,
        )
