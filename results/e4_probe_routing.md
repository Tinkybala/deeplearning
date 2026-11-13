# E4: degradation-aware routing between the clean-only and clean+diff probes

Mean over the 7 held-out generators (worst generator in brackets). Plain rows use the threshold from the clean seen-generator validation split, as in the main runs. **+recal** re-picks only the threshold on the six seen generators' validation images degraded like the test condition. **router** also picks, per condition, which score to use (mlp_clean+diff, logreg_clean or their logit-average **fusion**) by balanced accuracy on those degraded seen-generator images; it needs the degradation type at test time but never sees the held-out generator.

## Balanced accuracy

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff | **85.3%** [76.6] | 74.7% [64.2] | 81.5% [73.7] | 74.1% [68.2] | 61.6% [56.5] | 68.1% [56.5] | 74.2% |
| logreg_clean | 80.5% [68.6] | 74.8% [62.4] | 77.9% [66.8] | 73.1% [64.7] | 63.8% [58.9] | 70.1% [56.1] | 73.4% |
| fusion | 83.5% [73.2] | 76.5% [64.6] | 80.3% [69.7] | 74.0% [67.3] | 63.8% [58.5] | 70.7% [56.1] | 74.8% |
| mlp_clean+diff +recal | 85.1% [76.2] | **77.3%** [68.2] | **82.0%** [74.2] | 74.3% [67.2] | 61.9% [55.3] | 68.1% [57.8] | 74.8% |
| logreg_clean +recal | 80.3% [69.9] | 75.0% [63.7] | 78.6% [68.6] | 73.5% [63.6] | 63.9% [58.1] | 70.4% [59.3] | 73.6% |
| fusion +recal | 83.5% [72.3] | 77.1% [66.1] | 80.8% [70.2] | **74.8%** [64.9] | **65.1%** [57.9] | **71.1%** [58.0] | 75.4% |
| router | 85.1% [76.2] | 76.3% [66.1] | **82.0%** [74.2] | **74.8%** [64.9] | 64.6% [57.9] | **71.1%** [58.0] | 75.7% |

## AUROC

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff | **0.927** [0.844] | **0.854** [0.756] | **0.901** [0.829] | 0.823 [0.736] | 0.669 [0.585] | 0.750 [0.618] | 0.821 |
| logreg_clean | 0.883 [0.768] | 0.827 [0.694] | 0.864 [0.742] | 0.811 [0.700] | 0.697 [0.609] | 0.774 [0.622] | 0.809 |
| fusion | 0.913 [0.820] | 0.853 [0.736] | 0.890 [0.784] | **0.833** [0.724] | **0.705** [0.611] | **0.785** [0.628] | 0.830 |

## Mean balanced-accuracy difference vs `mlp_clean+diff` (pp), 7-generator average, paired bootstrap (2000 replicates per holdout)

| Method − mlp_clean+diff | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| fusion | -1.8* [-2.5, -1.1] | +1.7* [+0.9, +2.5] | -1.3* [-2.0, -0.5] | -0.1 [-0.9, +0.8] | +2.3* [+1.3, +3.2] | +2.6* [+1.7, +3.5] |
| router | -0.2 [-0.6, +0.1] | +1.5* [+0.7, +2.3] | +0.5* [+0.0, +1.0] | +0.7 [-0.2, +1.6] | +3.1* [+2.0, +4.1] | +3.1* [+2.2, +4.0] |
| mlp_clean+diff +recal | -0.2 [-0.6, +0.1] | +2.5* [+1.8, +3.2] | +0.5* [+0.0, +1.0] | +0.2 [-0.4, +0.8] | +0.3 [-0.4, +1.1] | +0.0 [-0.4, +0.5] |

## Mean balanced-accuracy difference vs `logreg_clean` (pp), 7-generator average, paired bootstrap (2000 replicates per holdout)

| Method − logreg_clean | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| fusion | +3.0* [+2.4, +3.6] | +1.6* [+1.0, +2.3] | +2.3* [+1.7, +2.9] | +1.0* [+0.3, +1.6] | +0.0 [-0.7, +0.8] | +0.5 [-0.2, +1.3] |
| router | +4.6* [+3.7, +5.5] | +1.5* [+0.6, +2.3] | +4.1* [+3.2, +5.0] | +1.7* [+0.9, +2.5] | +0.8* [+0.0, +1.7] | +1.0* [+0.3, +1.7] |
| logreg_clean +recal | -0.2 [-0.6, +0.3] | +0.2 [-0.2, +0.7] | +0.7* [+0.2, +1.2] | +0.4 [-0.3, +1.0] | +0.1 [-0.7, +0.8] | +0.2 [-0.4, +0.9] |

## Score chosen by the router (on degraded seen-generator validation)

| Held-out generator | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---|---|---|---|---|---|
| BigGAN | MLP | fusion | MLP | fusion | LR | fusion |
| VQDM | MLP | MLP | MLP | fusion | fusion | fusion |
| SD v1.5 | MLP | fusion | MLP | fusion | fusion | fusion |
| Wukong | MLP | MLP | MLP | fusion | fusion | fusion |
| ADM | MLP | MLP | MLP | fusion | fusion | fusion |
| GLIDE | MLP | MLP | MLP | fusion | fusion | fusion |
| Midjourney | MLP | fusion | MLP | fusion | fusion | fusion |
