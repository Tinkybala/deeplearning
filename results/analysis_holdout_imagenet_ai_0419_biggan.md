# Failure analysis: held-out `imagenet_ai_0419_biggan`

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
| dinov2_rigid | 81% / 97% | 66% / 99% | 65% / 96% | 29% / 98% | 1% / 100% | 26% / 98% |
| dinov2_logreg_clean | 88% / 95% | 86% / 74% | 85% / 93% | 77% / 87% | 59% / 74% | 87% / 73% |
| dinov2_logreg_clean_diff | 86% / 93% | 88% / 66% | 80% / 92% | 68% / 87% | 52% / 68% | 79% / 70% |
| dinov2_mlp_clean_diff | 89% / 96% | 94% / 75% | 86% / 95% | 74% / 89% | 52% / 79% | 81% / 79% |

## 3. Per-image error overlap: `dinov2_rigid` vs `dinov2_mlp_clean_diff`

| Condition | Both correct | Only `dinov2_rigid` wrong | Only `dinov2_mlp_clean_diff` wrong | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 845 | 83 | 49 | 23 |
| JPEG Q50 | 689 | 157 | 135 | 19 |
| Resize 128 | 752 | 155 | 52 | 41 |
| Resize 64 | 556 | 260 | 76 | 108 |
| Resize 32 | 400 | 257 | 105 | 238 |
| 128 + JPEG 50 | 509 | 289 | 112 | 90 |

## Figures

- `reports/figures/imagenet_ai_0419_biggan_class_recall.png`
- `reports/figures/imagenet_ai_0419_biggan_score_shift.png`
- `reports/figures/imagenet_ai_0419_biggan_failures_dinov2_mlp_clean_diff.png`
