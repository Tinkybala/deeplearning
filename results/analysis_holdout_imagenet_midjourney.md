# Failure analysis: held-out `imagenet_midjourney`

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
| dinov2_rigid | 85% / 48% | 76% / 44% | 74% / 48% | 41% / 66% | 2% / 97% | 38% / 59% |
| dinov2_logreg_clean | 89% / 48% | 85% / 39% | 86% / 48% | 75% / 54% | 54% / 64% | 86% / 26% |
| dinov2_logreg_clean_diff | 90% / 59% | 95% / 25% | 86% / 55% | 73% / 54% | 52% / 55% | 82% / 25% |
| dinov2_mlp_clean_diff | 89% / 67% | 94% / 34% | 88% / 59% | 79% / 58% | 53% / 60% | 83% / 30% |

## 3. Per-image error overlap: `dinov2_rigid` vs `dinov2_mlp_clean_diff`

| Condition | Both correct | Only `dinov2_rigid` wrong | Only `dinov2_mlp_clean_diff` wrong | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 598 | 183 | 68 | 151 |
| JPEG Q50 | 460 | 182 | 140 | 218 |
| Resize 128 | 524 | 213 | 89 | 174 |
| Resize 64 | 386 | 296 | 145 | 173 |
| Resize 32 | 304 | 261 | 193 | 242 |
| 128 + JPEG 50 | 281 | 284 | 202 | 233 |

## Figures

- `reports/figures/imagenet_midjourney_class_recall.png`
- `reports/figures/imagenet_midjourney_score_shift.png`
- `reports/figures/imagenet_midjourney_failures_dinov2_mlp_clean_diff.png`
