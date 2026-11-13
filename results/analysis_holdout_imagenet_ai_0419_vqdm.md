# Failure analysis: held-out `imagenet_ai_0419_vqdm`

## 1. Format and resolution confound in GenImage

First 100 training images per folder. A luminance DC step of 1–2 means JPEG quality ≈ 95–100.

| Generator | Class | Formats | Shorter side (median, range) | JPEG luminance DC quant. step (median) |
|---|---|---|---|---:|
| imagenet_ai_0419_biggan | real | JPEG 100 | 375 (98–700) | 1 |
| imagenet_ai_0419_biggan | generated | PNG 100 | 128 (128–128) | – |
| imagenet_ai_0419_vqdm | real | JPEG 100 | 352 (98–1200) | 1 |
| imagenet_ai_0419_vqdm | generated | PNG 100 | 256 (256–256) | – |
| imagenet_ai_0424_sdv5 | real | JPEG 100 | 374 (139–695) | 1 |
| imagenet_ai_0424_sdv5 | generated | PNG 100 | 512 (512–512) | – |
| imagenet_ai_0424_wukong | real | JPEG 100 | 375 (161–1411) | 1 |
| imagenet_ai_0424_wukong | generated | PNG 100 | 512 (512–512) | – |
| imagenet_ai_0508_adm | real | JPEG 100 | 350 (110–1050) | 1 |
| imagenet_ai_0508_adm | generated | PNG 100 | 256 (256–256) | – |
| imagenet_glide | real | JPEG 100 | 375 (160–1065) | 1 |
| imagenet_glide | generated | PNG 100 | 256 (256–256) | – |
| imagenet_midjourney | real | JPEG 100 | 375 (170–1080) | 1 |
| imagenet_midjourney | generated | PNG 100 | 1024 (1024–1024) | – |

## 2. Recall of each class (real / generated) at the validation-calibrated threshold

| Run | Clean real / gen. | JPEG Q50 real / gen. | Resize 128 real / gen. | Resize 64 real / gen. | Resize 32 real / gen. | 128 + JPEG 50 real / gen. |
|---|---:|---:|---:|---:|---:|---:|
| dinov2_rigid | 82% / 79% | 67% / 91% | 67% / 96% | 28% / 99% | 1% / 100% | 23% / 100% |
| dinov2_logreg_clean | 86% / 84% | 82% / 83% | 79% / 86% | 69% / 85% | 56% / 78% | 83% / 72% |
| dinov2_logreg_clean_diff | 89% / 81% | 92% / 67% | 79% / 82% | 68% / 81% | 52% / 72% | 78% / 60% |
| dinov2_mlp_clean_diff | 88% / 84% | 92% / 68% | 80% / 90% | 64% / 88% | 41% / 85% | 76% / 73% |

## 3. Per-image error overlap: `dinov2_rigid` vs `dinov2_mlp_clean_diff`

| Condition | Both correct | Only `dinov2_rigid` wrong | Only `dinov2_mlp_clean_diff` wrong | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 710 | 151 | 92 | 47 |
| JPEG Q50 | 628 | 170 | 165 | 37 |
| Resize 128 | 722 | 129 | 90 | 59 |
| Resize 64 | 543 | 219 | 96 | 142 |
| Resize 32 | 429 | 201 | 75 | 295 |
| 128 + JPEG 50 | 464 | 282 | 150 | 104 |

## Figures

- `reports/figures/imagenet_ai_0419_vqdm_class_recall.png`
- `reports/figures/imagenet_ai_0419_vqdm_score_shift.png`
- `reports/figures/imagenet_ai_0419_vqdm_failures_dinov2_mlp_clean_diff.png`
