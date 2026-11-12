"""Train and evaluate a ResNet-18 with one ImageNet-AI generator held out.

The held-out generator is excluded from both training and validation. The model
is fine-tuned on the other generators, the decision threshold is selected
without held-out images, and final results are reported on the held-out split.

Windows GPU run from the project root (after installing CUDA-enabled PyTorch):
    .venv\\Scripts\\python.exe scripts\\train_resnet_generalization.py --held-out imagenet_glide
To repeat only the held-out evaluation from the saved checkpoint, add --eval-only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "image" / "training" / "GenImage"
DEFAULT_TEST_DATA_ROOT = PROJECT_ROOT / "dataset" / "image" / "test" / "GenImage"
GENERATORS = [
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
    "imagenet_midjourney",
]


def portable_root(path: Path) -> str:
    """Describe a configured data root without recording machine-local paths."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def resize_shorter_side(image: Image.Image, side: int) -> Image.Image:
    """Resize while preserving aspect ratio, setting the shorter side to side."""
    width, height = image.size
    if width <= height:
        size = (side, round(height * side / width))
    else:
        size = (round(width * side / height), side)
    return image.resize(size, Image.Resampling.BICUBIC)


class RandomImageArtifacts:
    """Apply selected random compression/resizing artifacts to train images."""

    def __init__(self, use_jpeg: bool, use_resize: bool):
        self.use_jpeg = use_jpeg
        self.use_resize = use_resize

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.use_jpeg and random.random() < 0.5:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(30, 95))
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        if self.use_resize and random.random() < 0.5:
            side = random.choice((32, 64, 128))
            image = resize_shorter_side(image, side)
            image = resize_shorter_side(image, 256)
        return image


class ImageRecords(Dataset):
    def __init__(self, records, transform, condition="clean"):
        self.records = records
        self.transform = transform
        self.condition = condition

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label, generator = self.records[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.condition == "jpeg_q50":
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=50)
                buffer.seek(0)
                image = Image.open(buffer).convert("RGB")
            elif self.condition.startswith("resize_jpeg_"):
                side = int(self.condition.split("_")[2])
                image = resize_shorter_side(image, side)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=50)
                buffer.seek(0)
                image = Image.open(buffer).convert("RGB")
                image = resize_shorter_side(image, 256)
            elif self.condition.startswith("resize_"):
                side = int(self.condition.split("_")[1])
                image = resize_shorter_side(image, side)
                image = resize_shorter_side(image, 256)
            return self.transform(image), label, generator, str(path)


def files_for(root: Path, generator: str, split: str, category: str) -> list[Path]:
    folder = root / generator / split / category
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected dataset folder: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file())
    if not files:
        raise RuntimeError(f"No images found under {folder}")
    return files


def records_for(root: Path, generators: list[str], split: str, real_paths=None):
    records = []
    # The ImageNet real images are shared across generator folders. Read only
    # one copy to avoid duplicate real samples in training and validation.
    if real_paths is None:
        real_paths = files_for(root, generators[0], split, "nature")
    records.extend((p, 1, "real") for p in real_paths)
    for generator in generators:
        records.extend((p, 0, generator) for p in files_for(root, generator, split, "ai"))
    return records


def make_model(pretrained: bool, device: torch.device):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def get_scores(model, loader, device):
    model.eval()
    y_true, p_real, details = [], [], []
    with torch.inference_mode():
        for images, labels, groups, paths in tqdm(loader, desc="Scoring", leave=False):
            logits = model(images.to(device))
            prob = logits.softmax(dim=1)[:, 1].cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            p_real.extend(prob.tolist())
            details.extend(zip(groups, paths, labels.numpy().tolist(), prob.tolist()))
    return np.asarray(y_true), np.asarray(p_real), details


def best_threshold(labels, scores):
    candidates = np.unique(scores)
    # Average the observed score cutoffs to avoid choosing a threshold outside
    # the useful range while still optimizing balanced accuracy.
    best = (0.5, -1.0)
    for threshold in candidates:
        pred = scores >= threshold
        value = balanced_accuracy_score(labels, pred)
        if value > best[1]:
            best = (float(threshold), float(value))
    return best


def summarize(labels, scores, threshold):
    predicted_real = scores >= threshold
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predicted_real)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_real)),
        "auroc_real_positive": float(roc_auc_score(labels, scores)),
        "fake_f1": float(f1_score(1 - labels, 1 - predicted_real, zero_division=0)),
        "threshold": float(threshold),
        "real_correct": int(((labels == 1) & predicted_real).sum()),
        "real_total": int((labels == 1).sum()),
        "fake_correct": int(((labels == 0) & ~predicted_real).sum()),
        "fake_total": int((labels == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="GenImage training/validation root")
    parser.add_argument("--test-data-root", type=Path, default=DEFAULT_TEST_DATA_ROOT,
                        help="Separate held-out test root")
    parser.add_argument("--test-split", default="test",
                        help="Split directory under the held-out generator in --test-data-root")
    parser.add_argument("--held-out", default="imagenet_glide", choices=GENERATORS)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Train from random initialization")
    parser.add_argument("--eval-only", action="store_true", help="Reuse an existing best_model.pt and only run held-out tests")
    parser.add_argument("--robust-augmentation", action="store_true",
                        help="Alias for --augmentation both")
    parser.add_argument("--augmentation", choices=("none", "jpeg", "resize", "both"), default="none",
                        help="Train-time artifact augmentation ablation")
    args = parser.parse_args()
    for key in ("data_root", "test_data_root"):
        value = getattr(args, key).expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        setattr(args, key, value.resolve())
    if args.robust_augmentation and args.augmentation != "none":
        parser.error("Use either --robust-augmentation or --augmentation, not both")
    augmentation = "both" if args.robust_augmentation else args.augmentation

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seen = [g for g in GENERATORS if g != args.held_out]
    # Reserve a portion of the ImageNet training photos for threshold
    # calibration. The ImageNet validation photos are shared by every generator
    # folder, including the held-out test set, so using them for calibration
    # would expose test images during model selection.
    real_train_paths = files_for(args.data_root, seen[0], "train", "nature")
    random.Random(args.seed).shuffle(real_train_paths)
    calibration_count = max(1, len(real_train_paths) // 5)
    calibration_real = real_train_paths[:calibration_count]
    fitting_real = real_train_paths[calibration_count:]
    train_records = records_for(args.data_root, seen, "train", real_paths=fitting_real)
    val_records = records_for(args.data_root, seen, "val", real_paths=calibration_real)
    test_records = records_for(args.test_data_root, [args.held_out], args.test_split)

    train_steps = [transforms.Resize(256)]
    if augmentation != "none":
        train_steps.append(RandomImageArtifacts(
            use_jpeg=augmentation in ("jpeg", "both"),
            use_resize=augmentation in ("resize", "both"),
        ))
    train_steps.extend([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=ResNet18_Weights.DEFAULT.transforms().mean,
                             std=ResNet18_Weights.DEFAULT.transforms().std),
    ])
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=ResNet18_Weights.DEFAULT.transforms().mean,
                             std=ResNet18_Weights.DEFAULT.transforms().std),
    ])

    pin = device.type == "cuda"
    train_loader = DataLoader(ImageRecords(train_records, train_transform), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(ImageRecords(val_records, eval_transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=pin)
    resnet_model_root = PROJECT_ROOT / "model" / "ResNet" / "Frozen Resnet model"
    torch.hub.set_dir(str(resnet_model_root / "torch_hub"))
    model = make_model(not args.no_pretrained and not args.eval_only, device)
    run_names = {
        "none": f"resnet_holdout_{args.held_out}",
        "jpeg": f"resnet_jpeg_holdout_{args.held_out}",
        "resize": f"resnet_resize_holdout_{args.held_out}",
        "both": f"resnet_robust_holdout_{args.held_out}",
    }
    run_name = run_names[augmentation]
    out_dir = PROJECT_ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (resnet_model_root / "checkpoints" / run_name / "best_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if args.eval_only:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint found for --eval-only: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val = checkpoint.get("best_val_balanced_accuracy")
        if best_val is None:
            history_path = out_dir / "training_history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                best_val = max(row["val_balanced_accuracy"] for row in history)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        labels_for_training = np.asarray([label for _, label, _ in train_records])
        class_counts = np.bincount(labels_for_training, minlength=2)
        class_weights = torch.tensor(len(labels_for_training) / (2.0 * class_counts),
                                     dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        history = []
        best_val = -1.0
        for epoch in range(args.epochs):
            model.train()
            loss_sum, correct, count = 0.0, 0, 0
            for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                count += labels.size(0)
            scheduler.step()

            y_val, p_val, _ = get_scores(model, val_loader, device)
            threshold, val_balanced = best_threshold(y_val, p_val)
            row = {"epoch": epoch + 1, "train_loss": loss_sum / count,
                   "train_accuracy": correct / count, "val_balanced_accuracy": val_balanced,
                   "val_threshold": threshold}
            history.append(row)
            print(json.dumps(row))
            if val_balanced > best_val:
                best_val = val_balanced
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                            "threshold": threshold, "held_out": args.held_out, "seen": seen,
                            "seed": args.seed, "pretrained": not args.no_pretrained,
                            "best_val_balanced_accuracy": best_val}, checkpoint_path)
            with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = checkpoint["threshold"]

    conditions = ["clean", "jpeg_q50", "resize_128", "resize_64", "resize_32", "resize_jpeg_128"]
    summaries = {}
    all_details = []
    for condition in conditions:
        test_loader = DataLoader(ImageRecords(test_records, eval_transform, condition),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=pin)
        labels, scores, details = get_scores(model, test_loader, device)
        summaries[condition] = summarize(labels, scores, threshold)
        all_details.extend((condition, *record) for record in details)
        print(f"Test {condition}: {json.dumps(summaries[condition])}")

    with open(out_dir / "test_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "generator", "path", "label", "p_real", "prediction"])
        for condition, group, path, label, score in all_details:
            # Keep score files portable and avoid leaking the machine-local path.
            source_path = Path(path)
            try:
                portable_path = source_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                portable_path = source_path.relative_to(args.test_data_root.parent).as_posix()
            writer.writerow([condition, group, portable_path, "real" if label == 1 else "fake",
                             f"{score:.7f}", "real" if score >= threshold else "fake"])
    elapsed_key = "evaluation_seconds" if args.eval_only else "training_and_evaluation_seconds"
    result = {
        "model": "ResNet-18 fine-tuned from ImageNet pretrained weights" if not args.no_pretrained else "ResNet-18 from random initialization",
        "training_data_root": portable_root(args.data_root),
        "test_data_root": portable_root(args.test_data_root),
        "test_split": args.test_split,
        "held_out_generator": args.held_out,
        "seen_generators": seen,
        "train_images": len(train_records),
        "validation_images": len(val_records),
        "test_images": len(test_records),
        "best_epoch": checkpoint["epoch"],
        "training_augmentation": {
            "none": "random crop and flip only",
            "jpeg": "random JPEG quality 30-95 with p=0.5",
            "resize": "random downsampling to 32/64/128 px with p=0.5",
            "both": "random JPEG quality 30-95 and downsampling to 32/64/128 px, each p=0.5",
        }[augmentation],
        "threshold_selected_on_seen_validation": threshold,
        "seen_validation_balanced_accuracy": best_val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        elapsed_key: round(time.time() - start, 1),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "test_metrics": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_path} and results to {out_dir}")


if __name__ == "__main__":
    main()
