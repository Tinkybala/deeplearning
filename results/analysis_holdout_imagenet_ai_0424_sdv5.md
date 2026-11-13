# Failure analysis: held-out `imagenet_ai_0424_sdv5`

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
| dinov2_rigid | 77% / 53% | 61% / 53% | 61% / 52% | 26% / 73% | 0% / 99% | 21% / 74% |
| dinov2_logreg_clean | 77% / 83% | 70% / 76% | 71% / 83% | 58% / 82% | 42% / 84% | 68% / 63% |
| dinov2_logreg_clean_diff | 89% / 75% | 92% / 46% | 82% / 69% | 69% / 65% | 51% / 60% | 77% / 41% |
| dinov2_mlp_clean_diff | 83% / 88% | 90% / 64% | 75% / 81% | 66% / 76% | 44% / 74% | 71% / 57% |

## 3. Per-image error overlap: `dinov2_rigid` vs `dinov2_mlp_clean_diff`

| Condition | Both correct | Only `dinov2_rigid` wrong | Only `dinov2_mlp_clean_diff` wrong | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 589 | 267 | 60 | 84 |
| JPEG Q50 | 464 | 304 | 108 | 124 |
| Resize 128 | 484 | 294 | 83 | 139 |
| Resize 64 | 393 | 315 | 102 | 190 |
| Resize 32 | 368 | 225 | 128 | 279 |
| 128 + JPEG 50 | 315 | 321 | 160 | 204 |

## Figures

- `reports/figures/imagenet_ai_0424_sdv5_class_recall.png`
- `reports/figures/imagenet_ai_0424_sdv5_score_shift.png`
- `reports/figures/imagenet_ai_0424_sdv5_failures_dinov2_mlp_clean_diff.png`
