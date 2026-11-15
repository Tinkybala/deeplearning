"""Validate and summarize the seven matched-format ResNet holdout runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATORS = [
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
    "imagenet_midjourney",
]
CONDITIONS = ["clean", "jpeg_q50", "resize_128", "resize_64", "resize_32", "resize_jpeg_128"]
RUN_PREFIX = "resnet_rigid_format_holdout_"


def main() -> None:
    rows = []
    for generator in GENERATORS:
        run_name = f"{RUN_PREFIX}{generator}"
        result_dir = PROJECT_ROOT / "results" / run_name
        summary_path = result_dir / "summary.json"
        scores_path = result_dir / "test_scores.csv"
        checkpoint_path = (
            PROJECT_ROOT
            / "model"
            / "ResNet"
            / "Frozen Resnet model"
            / "checkpoints"
            / run_name
            / "best_model.pt"
        )
        if not summary_path.is_file() or not scores_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing summary, score CSV, or checkpoint for {generator}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["held_out_generator"] != generator:
            raise ValueError(f"Wrong held-out generator recorded in {summary_path}")
        if summary["evaluation_preprocessing"] != "squash224":
            raise ValueError(f"Unexpected evaluation format in {summary_path}")
        if summary["test_images"] != 1000 or set(summary["test_metrics"]) != set(CONDITIONS):
            raise ValueError(f"Unexpected test size or conditions in {summary_path}")
        expected_seen = set(GENERATORS) - {generator}
        if set(summary["seen_generators"]) != expected_seen:
            raise ValueError(f"Held-out generator leaked into seen set for {generator}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("held_out") != generator:
            raise ValueError(f"Checkpoint metadata mismatch for {generator}")

        condition_counts: dict[str, Counter[str]] = {condition: Counter() for condition in CONDITIONS}
        source_paths: set[str] = set()
        with scores_path.open(newline="", encoding="utf-8") as score_file:
            reader = csv.DictReader(score_file)
            for record in reader:
                condition = record["condition"]
                if condition not in condition_counts:
                    raise ValueError(f"Unexpected condition {condition!r} in {scores_path}")
                condition_counts[condition][record["label"]] += 1
                source_paths.add(record["path"])
        expected_classes = Counter({"real": 500, "fake": 500})
        if any(condition_counts[c] != expected_classes for c in CONDITIONS):
            raise ValueError(f"Test CSV class counts are not 500/500 per condition for {generator}")
        missing_paths = [path for path in source_paths if not (PROJECT_ROOT / path).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing image referenced by {scores_path}: {missing_paths[0]}")

        clean = summary["test_metrics"]["clean"]
        rows.append(
            {
                "generator": generator,
                "balanced_accuracy": clean["balanced_accuracy"],
                "auroc": clean["auroc_real_positive"],
                "accuracy": clean["accuracy"],
                "best_epoch": summary["best_epoch"],
                "seen_validation_balanced_accuracy": summary["seen_validation_balanced_accuracy"],
                "threshold": summary["threshold_selected_on_seen_validation"],
                "test_images": summary["test_images"],
                "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                "scores_csv": scores_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    mean_balanced_accuracy = sum(row["balanced_accuracy"] for row in rows) / len(rows)
    mean_auroc = sum(row["auroc"] for row in rows) / len(rows)
    result = {
        "experiment": "ResNet-18 leave-one-generator-out, DINOv2-matched squash224 evaluation",
        "seed": 42,
        "epochs": 8,
        "test_images_per_generator": 1000,
        "real_images_per_generator": 500,
        "fake_images_per_generator": 500,
        "generator_macro_mean_balanced_accuracy": mean_balanced_accuracy,
        "generator_macro_mean_auroc": mean_auroc,
        "note": "Real validation images are shared across GenImage generator folders, so the macro mean is descriptive across generator holdouts, not seven independent real-image samples.",
        "runs": rows,
    }

    csv_path = PROJECT_ROOT / "results" / "resnet_rigid_format_lodo_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path = PROJECT_ROOT / "results" / "resnet_rigid_format_lodo_summary.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    labels = [
        {
            "imagenet_ai_0419_biggan": "BigGAN",
            "imagenet_ai_0419_vqdm": "VQDM",
            "imagenet_ai_0424_sdv5": "Stable Diffusion v5",
            "imagenet_ai_0424_wukong": "Wukong",
            "imagenet_ai_0508_adm": "ADM",
            "imagenet_glide": "GLIDE",
            "imagenet_midjourney": "Midjourney",
        }[row["generator"]]
        for row in rows
    ]
    values = [row["balanced_accuracy"] * 100 for row in rows]
    colors = ["#376996" if value >= 50 else "#bc4749" for value in values]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(50, color="#333333", linestyle="--", linewidth=1, label="Chance (balanced accuracy)")
    ax.axhline(mean_balanced_accuracy * 100, color="#d17a22", linestyle=":", linewidth=1.5,
               label=f"Generator macro mean ({mean_balanced_accuracy:.1%})")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("ResNet-18 clean test by held-out GenImage generator\n224×224 resize matched to DINOv2/RIGID")
    ax.tick_params(axis="x", labelrotation=25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:.1f}%", ha="center", fontsize=9)
    ax.legend(frameon=False)
    figure_path = PROJECT_ROOT / "reports" / "figures" / "resnet_lodo_squash224_clean.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(json.dumps({
        "runs_validated": len(rows),
        "macro_balanced_accuracy": mean_balanced_accuracy,
        "macro_auroc": mean_auroc,
        "summary_csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
        "summary_json": json_path.relative_to(PROJECT_ROOT).as_posix(),
        "figure": figure_path.relative_to(PROJECT_ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
