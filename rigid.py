"""RIGID: training-free AI-generated image detection.

Port of the detector in IBM's RIGID notebook (https://github.com/IBM/RIGID).

Method: embed an image with a frozen vision foundation model (DINOv2 ViT-L/14),
embed a slightly noise-perturbed copy, and take the cosine similarity of the two
embeddings. Real images are more robust to the noise, so they score HIGHER;
AI-generated images score LOWER.

The score is all the paper defines. Turning it into a real/fake label needs a
threshold, which you should calibrate on images resembling the ones you test
(see `calibrate_threshold`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
PROJECT_ROOT = Path(__file__).resolve().parent
DINO_MODEL_ROOT = PROJECT_ROOT / "model" / "DINOv2" / "Frozen DINO model"

# Backbones available from torch.hub 'facebookresearch/dinov2'. The paper uses vitl14.
DINOV2_MODELS = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")


def get_device(device: str | None = None) -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU, unless a device is given."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(image_size: int = 224) -> transforms.Compose:
    """Same preprocessing as the notebook: resize, to tensor, ImageNet normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])


def load_model(name: str = "dinov2_vitl14", device: str | None = None) -> torch.nn.Module:
    """Load frozen DINOv2 weights, cached under the project's model directory."""
    if name not in DINOV2_MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {DINOV2_MODELS}")
    torch.hub.set_dir(str(DINO_MODEL_ROOT / "torch_hub"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is not available")  # optional speedup only
        model = torch.hub.load("facebookresearch/dinov2", name)
    model.eval().requires_grad_(False)
    return model.to(get_device(device))


@dataclass
class RIGIDResult:
    path: str | None
    similarity: float           # mean cosine similarity over noise draws (higher = more "real")
    similarity_std: float       # spread across noise draws
    threshold: float | None
    label: str | None           # "real" / "ai-generated", or None if no threshold

    def __str__(self) -> str:
        s = f"similarity = {self.similarity:.4f} (± {self.similarity_std:.4f})"
        if self.label is not None:
            s += f"  ->  {self.label.upper()} (threshold {self.threshold:.4f})"
        return s


class RIGIDDetector:
    """RIGID detector.

    Args:
        lamb: noise intensity, added in normalized-pixel space (paper/notebook: 0.05).
        model_name: DINOv2 backbone from torch.hub.
        device: "cuda", "mps", "cpu", or None for auto.
        image_size: input resolution (224 in the notebook; must be a multiple of 14).
        threshold: optional similarity cutoff; scores below it are labelled AI-generated.
        seed: seed for the noise generator, so repeated runs give the same score.
    """

    def __init__(
        self,
        lamb: float = 0.05,
        model_name: str = "dinov2_vitl14",
        device: str | None = None,
        image_size: int = 224,
        threshold: float | None = None,
        seed: int | None = 100,
    ):
        self.lamb = lamb
        self.device = get_device(device)
        self.model = load_model(model_name, self.device)
        self.transform = build_transform(image_size)
        self.threshold = threshold
        self.seed = seed

    # ---- core method (identical to RIGID_Detector.calculate_sim in the notebook) ----
    @torch.no_grad()
    def calculate_sim(self, data: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Cosine similarity between features of `data` and of `data + lamb * N(0, 1)`.

        `data` is a preprocessed batch (B, 3, H, W) already on `self.device`.
        """
        features = self.model(data)
        noise = torch.randn(data.shape, generator=generator, device="cpu").to(data.device)
        trans_features = self.model(data + noise * self.lamb)
        return F.cosine_similarity(features, trans_features, dim=-1)

    @torch.no_grad()
    def score_batch(self, data: torch.Tensor, n_noise: int = 1,
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Per-image similarity for a preprocessed batch, averaged over `n_noise` draws.

        With n_noise=1 this is exactly the notebook's `calculate_sim`. Returns (B,) on CPU.
        """
        data = data.to(self.device)
        features = self.model(data)
        total = torch.zeros(data.shape[0], device=self.device)
        for _ in range(n_noise):
            noise = torch.randn(data.shape, generator=generator).to(self.device)
            total += F.cosine_similarity(features, self.model(data + noise * self.lamb), dim=-1)
        return (total / n_noise).cpu()

    # ---- convenience for single images ----
    def preprocess(self, image: str | Path | Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def score(self, image: str | Path | Image.Image, n_noise: int = 1) -> tuple[float, float]:
        """Return (mean, std) similarity over `n_noise` independent noise draws.

        The notebook uses one draw per image; averaging a few makes a single-image
        score less sensitive to the particular noise sample.
        """
        x = self.preprocess(image)
        gen = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        # Embed the clean image once, then each noisy copy in one batch.
        features = self.model(x)
        noise = torch.randn((n_noise, *x.shape[1:]), generator=gen).to(self.device)
        trans_features = self.model(x + noise * self.lamb)
        sims = F.cosine_similarity(features, trans_features, dim=-1)
        std = sims.std().item() if n_noise > 1 else 0.0
        return sims.mean().item(), std

    def detect(self, image: str | Path | Image.Image, n_noise: int = 1,
               threshold: float | None = None) -> RIGIDResult:
        """Score one image and, if a threshold is set, label it real / AI-generated."""
        threshold = self.threshold if threshold is None else threshold
        mean, std = self.score(image, n_noise=n_noise)
        label = None
        if threshold is not None:
            label = "real" if mean >= threshold else "ai-generated"
        path = None if isinstance(image, Image.Image) else str(image)
        return RIGIDResult(path, mean, std, threshold, label)


def calibrate_threshold(real_scores, fake_scores) -> float:
    """Threshold maximizing balanced accuracy between known-real and known-fake scores."""
    real = torch.as_tensor(list(real_scores), dtype=torch.float64)
    fake = torch.as_tensor(list(fake_scores), dtype=torch.float64)
    if len(real) == 0 or len(fake) == 0:
        raise ValueError("Need at least one real and one fake score")
    candidates = torch.cat([real, fake]).unique()
    best_t, best_acc = candidates[0].item(), -1.0
    for t in candidates:
        acc = 0.5 * ((real >= t).double().mean() + (fake < t).double().mean()).item()
        if acc > best_acc:
            best_t, best_acc = t.item(), acc
    return best_t
