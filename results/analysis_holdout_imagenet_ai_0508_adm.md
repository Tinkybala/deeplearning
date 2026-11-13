# Failure analysis: held-out `imagenet_ai_0508_adm`

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
| dinov2_rigid | 80% / 67% | 63% / 77% | 63% / 84% | 28% / 94% | 1% / 100% | 24% / 96% |
| dinov2_logreg_clean | 83% / 56% | 77% / 60% | 75% / 64% | 65% / 70% | 45% / 76% | 78% / 51% |
| dinov2_logreg_clean_diff | 87% / 61% | 89% / 38% | 75% / 72% | 62% / 74% | 48% / 71% | 81% / 42% |
| dinov2_mlp_clean_diff | 89% / 64% | 92% / 37% | 84% / 68% | 73% / 72% | 47% / 75% | 83% / 47% |

## 3. Per-image error overlap: `dinov2_rigid` vs `dinov2_mlp_clean_diff`

| Condition | Both correct | Only `dinov2_rigid` wrong | Only `dinov2_mlp_clean_diff` wrong | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 640 | 126 | 99 | 135 |
| JPEG Q50 | 460 | 182 | 239 | 119 |
| Resize 128 | 603 | 157 | 135 | 105 |
| Resize 64 | 464 | 258 | 145 | 133 |
| Resize 32 | 379 | 233 | 126 | 262 |
| 128 + JPEG 50 | 348 | 306 | 252 | 94 |

## Figures

- `reports/figures/imagenet_ai_0508_adm_class_recall.png`
- `reports/figures/imagenet_ai_0508_adm_score_shift.png`
- `reports/figures/imagenet_ai_0508_adm_failures_dinov2_mlp_clean_diff.png`
