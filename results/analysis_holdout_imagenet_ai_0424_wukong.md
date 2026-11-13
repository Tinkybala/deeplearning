# Failure analysis: held-out `imagenet_ai_0424_wukong`

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
| dinov2_rigid | 79% / 58% | 64% / 58% | 63% / 61% | 29% / 78% | 1% / 99% | 24% / 71% |
| dinov2_logreg_clean | 76% / 86% | 71% / 86% | 72% / 86% | 59% / 88% | 41% / 87% | 73% / 73% |
| dinov2_logreg_clean_diff | 84% / 85% | 87% / 62% | 75% / 83% | 60% / 80% | 44% / 77% | 70% / 60% |
| dinov2_mlp_clean_diff | 87% / 86% | 91% / 63% | 83% / 81% | 72% / 77% | 49% / 77% | 79% / 58% |

## 3. Per-image error overlap: `dinov2_rigid` vs `dinov2_mlp_clean_diff`

| Condition | Both correct | Only `dinov2_rigid` wrong | Only `dinov2_mlp_clean_diff` wrong | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 625 | 240 | 63 | 72 |
| JPEG Q50 | 495 | 274 | 111 | 120 |
| Resize 128 | 542 | 277 | 81 | 100 |
| Resize 64 | 450 | 292 | 87 | 171 |
| Resize 32 | 384 | 246 | 113 | 257 |
| 128 + JPEG 50 | 341 | 348 | 133 | 178 |

## Figures

- `reports/figures/imagenet_ai_0424_wukong_class_recall.png`
- `reports/figures/imagenet_ai_0424_wukong_score_shift.png`
- `reports/figures/imagenet_ai_0424_wukong_failures_dinov2_mlp_clean_diff.png`
