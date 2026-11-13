# Leave-one-generator-out summary

Held-out generators averaged (7): BigGAN, VQDM, SD v1.5, Wukong, ADM, GLIDE, Midjourney. Each holdout tests 500 real and 500 generated images per condition; thresholds are calibrated on the seen generators only.

## Clean balanced accuracy per held-out generator

| Held-out generator | dinov2_rigid | dinov2_logreg_clean | dinov2_logreg_clean_diff | dinov2_mlp_clean_diff |
|---|---:|---:|---:|---:|
| BigGAN | 89.4% | 91.5% | 89.2% | **92.8%** |
| VQDM | 80.2% | 85.4% | 84.8% | **86.1%** |
| SD v1.5 | 64.9% | 79.6% | 81.6% | **85.6%** |
| Wukong | 68.8% | 81.2% | 84.6% | **86.5%** |
| ADM | 73.9% | 69.8% | 74.4% | **76.6%** |
| GLIDE | 87.6% | 87.5% | **91.7%** | 91.5% |
| Midjourney | 66.6% | 68.6% | 74.9% | **78.1%** |

## Mean balanced accuracy over held-out generators [worst generator]

| Run | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| dinov2_rigid | 75.9% [64.9] | 69.1% [57.2] | 70.3% [56.7] | 57.3% [49.5] | 50.0% [49.6] | 54.4% [47.4] |
| dinov2_logreg_clean | 80.5% [68.6] | **74.8%** [62.4] | 77.9% [66.8] | 73.1% [64.7] | **63.8%** [58.9] | **70.1%** [56.1] |
| dinov2_logreg_clean_diff | 83.0% [74.4] | 71.5% [59.7] | 78.6% [70.4] | 70.2% [63.5] | 58.7% [53.7] | 64.4% [53.6] |
| dinov2_mlp_clean_diff | **85.3%** [76.6] | 74.7% [64.2] | **81.5%** [73.7] | **74.1%** [68.2] | 61.6% [56.5] | 68.1% [56.5] |

## Clean auroc (real positive) per held-out generator

| Held-out generator | dinov2_rigid | dinov2_logreg_clean | dinov2_logreg_clean_diff | dinov2_mlp_clean_diff |
|---|---:|---:|---:|---:|
| BigGAN | 0.976 | 0.974 | 0.958 | **0.983** |
| VQDM | 0.875 | 0.928 | 0.924 | **0.935** |
| SD v1.5 | 0.717 | 0.874 | 0.901 | **0.927** |
| Wukong | 0.760 | 0.898 | 0.918 | **0.942** |
| ADM | 0.807 | 0.787 | 0.817 | **0.844** |
| GLIDE | 0.958 | 0.954 | 0.978 | **0.982** |
| Midjourney | 0.782 | 0.768 | 0.841 | **0.879** |

## Mean auroc (real positive) over held-out generators [worst generator]

| Run | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| dinov2_rigid | 0.839 [0.717] | 0.770 [0.606] | 0.791 [0.591] | 0.692 [0.484] | 0.574 [0.396] | 0.655 [0.454] |
| dinov2_logreg_clean | 0.883 [0.768] | 0.827 [0.694] | 0.864 [0.742] | 0.811 [0.700] | **0.697** [0.609] | **0.774** [0.622] |
| dinov2_logreg_clean_diff | 0.905 [0.817] | 0.804 [0.694] | 0.869 [0.787] | 0.779 [0.689] | 0.621 [0.529] | 0.691 [0.548] |
| dinov2_mlp_clean_diff | **0.927** [0.844] | **0.854** [0.756] | **0.901** [0.829] | **0.823** [0.736] | 0.669 [0.585] | 0.750 [0.618] |
