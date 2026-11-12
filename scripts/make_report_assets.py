"""Keep ResNet result metadata portable and regenerate report figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "reports" / "figures"
RUNS = {
    "Baseline": "resnet_holdout_imagenet_glide",
    "JPEG only": "resnet_jpeg_holdout_imagenet_glide",
    "Resize only": "resnet_resize_holdout_imagenet_glide",
    "JPEG + resize": "resnet_robust_holdout_imagenet_glide",
}
CONDITIONS = [
    ("clean", "Clean"),
    ("jpeg_q50", "JPEG Q50"),
    ("resize_128", "Resize 128"),
    ("resize_64", "Resize 64"),
    ("resize_32", "Resize 32"),
    ("resize_jpeg_128", "Resize 128\n+ JPEG Q50"),
]
COLORS = ["#334155", "#2563eb", "#f59e0b", "#10b981"]


def load_and_sanitize() -> tuple[dict, dict]:
    summaries, histories = {}, {}
    for label, folder in RUNS.items():
        run_dir = RESULTS / folder
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("dataset", None)
        summary["training_data_root"] = "dataset/image/training/GenImage"
        summary["test_data_root"] = "dataset/image/test/GenImage"
        summary["test_split"] = "test"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        scores_path = run_dir / "test_scores.csv"
        with scores_path.open("r", newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            row["path"] = row["path"].replace("\\", "/")
        with scores_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=["condition", "generator", "path", "label", "p_real", "prediction"])
            writer.writeheader()
            writer.writerows(rows)

        summaries[label] = summary
        histories[label] = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    return summaries, histories


def plot_test_metrics(summaries: dict, metric: str, ylabel: str, filename: str) -> None:
    x = np.arange(len(CONDITIONS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.5, 6.6), constrained_layout=True)
    for run_index, (label, _) in enumerate(RUNS.items()):
        vals = [summaries[label]["test_metrics"][key][metric] for key, _ in CONDITIONS]
        ax.bar(x + (run_index - 1.5) * width, vals, width, label=label, color=COLORS[run_index])
    ax.set_xticks(x, [label for _, label in CONDITIONS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.4 if metric == "balanced_accuracy" else 0.45, 1.0)
    ax.set_title(f"Held-out GLIDE performance by test condition ({ylabel})")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(FIGURES / filename, dpi=180, facecolor="white")
    plt.close(fig)


def plot_training(histories: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for idx, (label, _) in enumerate(RUNS.items()):
        history = histories[label]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_accuracy"] for row in history], marker="o", color=COLORS[idx], label=label)
        axes[1].plot(epochs, [row["val_balanced_accuracy"] for row in history], marker="o", color=COLORS[idx], label=label)
    axes[0].set_title("Training accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Seen-generator validation balanced accuracy")
    axes[1].set_ylabel("Balanced accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 9))
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Training traces used for model selection", fontsize=14)
    fig.savefig(FIGURES / "training_curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summaries, histories = load_and_sanitize()
    plot_test_metrics(summaries, "balanced_accuracy", "Balanced accuracy", "test_balanced_accuracy.png")
    plot_test_metrics(summaries, "auroc_real_positive", "AUROC", "test_auroc.png")
    plot_training(histories)
    print(f"Updated portable summaries and score CSVs; wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
