# Leave-one-generator-out summary

Held-out generators averaged (6): BigGAN, VQDM, SD v1.5, Wukong, ADM, GLIDE. Not yet available for every run: Midjourney. Each holdout tests 500 real and 500 generated images per condition; thresholds are calibrated on the seen generators only.

## Clean balanced accuracy per held-out generator

| Held-out generator | dinov2_rigid | dinov2_logreg_clean | dinov2_logreg_clean_diff | dinov2_mlp_clean_diff |
|---|---:|---:|---:|---:|
| BigGAN | 89.4% | 91.5% | 89.2% | **92.8%** |
| VQDM | 80.2% | 85.4% | 84.8% | **86.1%** |
| SD v1.5 | 64.9% | 79.6% | 81.6% | **85.6%** |
| Wukong | 68.8% | 81.2% | 84.6% | **86.5%** |
| ADM | 73.9% | 69.8% | 74.4% | **76.6%** |
| GLIDE | 87.6% | 87.5% | **91.7%** | 91.5% |

## Mean balanced accuracy over held-out generators [worst generator]

| Run | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| dinov2_rigid | 77.5% [64.9] | 70.6% [57.2] | 71.8% [56.7] | 58.0% [49.5] | 50.1% [49.6] | 55.4% [47.4] |
| dinov2_logreg_clean | 82.5% [69.8] | **76.9%** [68.2] | 79.8% [69.2] | 74.5% [67.8] | **64.6%** [60.6] | **72.5%** [64.3] |
| dinov2_logreg_clean_diff | 84.4% [74.4] | 73.5% [63.5] | 80.0% [73.9] | 71.3% [66.9] | 59.6% [55.2] | 66.2% [59.2] |
| dinov2_mlp_clean_diff | **86.5%** [76.6] | 76.5% [64.2] | **82.8%** [76.0] | **75.1%** [70.8] | 62.4% [59.3] | 70.0% [63.6] |

## Clean auroc (real positive) per held-out generator

| Held-out generator | dinov2_rigid | dinov2_logreg_clean | dinov2_logreg_clean_diff | dinov2_mlp_clean_diff |
|---|---:|---:|---:|---:|
| BigGAN | 0.976 | 0.974 | 0.958 | **0.983** |
| VQDM | 0.875 | 0.928 | 0.924 | **0.935** |
| SD v1.5 | 0.717 | 0.874 | 0.901 | **0.927** |
| Wukong | 0.760 | 0.898 | 0.918 | **0.942** |
| ADM | 0.807 | 0.787 | 0.817 | **0.844** |
| GLIDE | 0.958 | 0.954 | 0.978 | **0.982** |

## Mean auroc (real positive) over held-out generators [worst generator]

| Run | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| dinov2_rigid | 0.849 [0.717] | 0.788 [0.606] | 0.807 [0.591] | 0.718 [0.484] | 0.595 [0.396] | 0.685 [0.454] |
| dinov2_logreg_clean | 0.902 [0.787] | 0.849 [0.746] | 0.884 [0.787] | 0.830 [0.752] | **0.712** [0.663] | **0.799** [0.717] |
| dinov2_logreg_clean_diff | 0.916 [0.817] | 0.822 [0.722] | 0.883 [0.816] | 0.794 [0.713] | 0.637 [0.566] | 0.715 [0.642] |
| dinov2_mlp_clean_diff | **0.936** [0.844] | **0.866** [0.756] | **0.913** [0.844] | **0.838** [0.773] | 0.683 [0.642] | 0.772 [0.701] |
