"""Leave-one-generator-out summary across every GenImage holdout that has results.

Reads results/<run>_holdout_<generator>/summary.json for each detector and writes
results/logo_summary.md with:

1. Clean balanced accuracy and AUROC per held-out generator (rows) and detector (columns).
2. Mean over held-out generators per test condition, with the worst generator in brackets.

Only generators for which every listed run is present are averaged, so the mean always
covers the same holdouts for every detector.

    python scripts/summarize_logo.py
    python scripts/summarize_logo.py --runs dinov2_rigid dinov2_logreg_clean resnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = [
    ("imagenet_ai_0419_biggan", "BigGAN"),
    ("imagenet_ai_0419_vqdm", "VQDM"),
    ("imagenet_ai_0424_sdv5", "SD v1.5"),
    ("imagenet_ai_0424_wukong", "Wukong"),
    ("imagenet_ai_0508_adm", "ADM"),
    ("imagenet_glide", "GLIDE"),
    ("imagenet_midjourney", "Midjourney"),
]
CONDITIONS = [("clean", "Clean"), ("jpeg_q50", "JPEG Q50"), ("resize_128", "Resize 128"),
              ("resize_64", "Resize 64"), ("resize_32", "Resize 32"), ("resize_jpeg_128", "128 + JPEG 50")]
METRICS = [("balanced_accuracy", "Balanced accuracy", lambda v: f"{v:.1%}"),
           ("auroc_real_positive", "AUROC (real positive)", lambda v: f"{v:.3f}")]


def load(run: str, generator: str) -> dict | None:
    path = ROOT / "results" / f"{run}_holdout_{generator}" / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return None if summary.get("max_images_per_group") else summary["test_metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+",
                        default=["dinov2_rigid", "dinov2_logreg_clean", "dinov2_logreg_clean_diff",
                                 "dinov2_mlp_clean_diff"])
    args = parser.parse_args()

    metrics = {(run, g): load(run, g) for run in args.runs for g, _ in GENERATORS}
    complete = [(g, label) for g, label in GENERATORS if all(metrics[(r, g)] for r in args.runs)]
    missing = [label for g, label in GENERATORS if (g, label) not in complete]
    if not complete:
        raise SystemExit("No held-out generator has results for every requested run")

    sections = ["# Leave-one-generator-out summary",
                f"Held-out generators averaged ({len(complete)}): {', '.join(l for _, l in complete)}."
                + (f" Not yet available for every run: {', '.join(missing)}." if missing else "")
                + " Each holdout tests 500 real and 500 generated images per condition; thresholds are"
                  " calibrated on the seen generators only."]

    for metric, title, fmt in METRICS:
        # 1. clean result per held-out generator
        lines = ["| Held-out generator | " + " | ".join(args.runs) + " |", "|---|" + "---:|" * len(args.runs)]
        for g, label in complete:
            vals = [metrics[(r, g)]["clean"][metric] for r in args.runs]
            cells = [f"**{fmt(v)}**" if v == max(vals) else fmt(v) for v in vals]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        sections += [f"## Clean {title.lower()} per held-out generator", "\n".join(lines)]

        # 2. mean over holdouts per condition, worst holdout in brackets
        lines = ["| Run | " + " | ".join(c for _, c in CONDITIONS) + " |", "|---|" + "---:|" * len(CONDITIONS)]
        means = {}
        for r in args.runs:
            for c, _ in CONDITIONS:
                vals = [metrics[(r, g)][c][metric] for g, _ in complete]
                means[(r, c)] = (sum(vals) / len(vals), min(vals))
        for r in args.runs:
            cells = []
            for c, _ in CONDITIONS:
                mean, worst = means[(r, c)]
                text = fmt(mean)
                if mean == max(means[(o, c)][0] for o in args.runs):
                    text = f"**{text}**"
                cells.append(f"{text} [{fmt(worst)}]".replace("%]", "]"))
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        sections += [f"## Mean {title.lower()} over held-out generators [worst generator]", "\n".join(lines)]

    text = "\n\n".join(sections) + "\n"
    out = ROOT / "results" / "logo_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
