# Failure analysis: held-out `imagenet_glide`

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
| resnet | 77% / 93% | 77% / 36% | 44% / 97% | 3% / 99% | 1% / 100% | 83% / 40% |
| resnet_jpeg | 76% / 92% | 75% / 77% | 23% / 97% | 7% / 95% | 2% / 99% | 37% / 75% |
| resnet_robust | 72% / 74% | 70% / 66% | 68% / 75% | 69% / 66% | 49% / 78% | 58% / 65% |
| dinov2_rigid | 80% / 95% | 67% / 82% | 65% / 88% | 27% / 87% | 0% / 99% | 20% / 88% |
| dinov2_logreg_clean | 84% / 91% | 79% / 79% | 76% / 86% | 70% / 83% | 54% / 79% | 81% / 68% |
| dinov2_mlp_clean_diff | 87% / 96% | 92% / 62% | 79% / 92% | 69% / 82% | 46% / 78% | 77% / 58% |

## 3. Per-image error overlap: `resnet` vs `dinov2_mlp_clean_diff`

| Condition | Both correct | Only `resnet` wrong | Only `dinov2_mlp_clean_diff` wrong | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 786 | 129 | 65 | 20 |
| JPEG Q50 | 475 | 292 | 92 | 141 |
| Resize 128 | 630 | 225 | 78 | 67 |
| Resize 64 | 414 | 341 | 96 | 149 |
| Resize 32 | 392 | 230 | 111 | 267 |
| 128 + JPEG 50 | 437 | 240 | 181 | 142 |

## Figures

- `reports/figures/imagenet_glide_class_recall.png`
- `reports/figures/imagenet_glide_score_shift.png`
- `reports/figures/imagenet_glide_failures_resnet.png`
- `reports/figures/imagenet_glide_failures_dinov2_mlp_clean_diff.png`
