# A3: router with a predicted degradation type

E4's router needs the degradation type of each test image. Here a multinomial logistic regression (features clean+diff, C=0.1) predicts it from the frozen DINOv2 embedding. It is trained on the six seen generators' validation images (real and generated) under the six conditions, the same images the router already uses to pick its score and threshold, so the held-out generator is still never seen. Each test image is then scored with the probe and threshold the router chose for its *predicted* condition. The per-image decision is the signed logit distance to that threshold, which also serves as the AUROC score. Mean over the 7 held-out generators (worst generator in brackets).

## Balanced accuracy

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff | 85.3% [76.6] | 74.7% [64.2] | 81.5% [73.7] | 74.1% [68.2] | 61.6% [56.5] | 68.1% [56.5] | 74.2% |
| logreg_clean | 80.5% [68.6] | 74.8% [62.4] | 77.9% [66.8] | 73.1% [64.7] | 63.8% [58.9] | 70.1% [56.1] | 73.4% |
| fusion | 83.5% [73.2] | 76.5% [64.6] | 80.3% [69.7] | 74.0% [67.3] | 63.8% [58.5] | 70.7% [56.1] | 74.8% |
| router (given type) | 85.1% [76.2] | 76.3% [66.1] | 82.0% [74.2] | 74.8% [64.9] | 64.6% [57.9] | 71.1% [58.0] | 75.7% |
| router (predicted type) | 85.0% [76.2] | 75.7% [62.0] | 81.6% [72.9] | 74.5% [65.1] | 64.8% [58.3] | 70.7% [57.8] | 75.4% |

## Balanced-accuracy differences (pp), 7-generator mean, paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| router (predicted type) − mlp_clean+diff | -0.3 [-0.7, +0.1] | +0.9* [+0.2, +1.7] | +0.1 [-0.4, +0.6] | +0.4 [-0.5, +1.3] | +3.2* [+2.2, +4.3] | +2.6* [+1.7, +3.5] |
| router (predicted type) − logreg_clean | +4.5* [+3.6, +5.4] | +0.8 [-0.0, +1.7] | +3.7* [+2.8, +4.6] | +1.4* [+0.6, +2.2] | +1.0* [+0.1, +1.8] | +0.5 [-0.2, +1.2] |
| router (predicted type) − router (given type) | -0.1 [-0.3, +0.2] | -0.6* [-1.0, -0.3] | -0.4* [-0.7, -0.1] | -0.3 [-0.6, +0.0] | +0.2* [+0.0, +0.3] | -0.5* [-0.8, -0.2] |

## Degradation-classifier confusion on the held-out generators' test images (7-generator mean; overall accuracy 83.0%)

| True condition | → Clean | → JPEG Q50 | → Resize 128 | → Resize 64 | → Resize 32 | → 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | **75%** | 7% | 15% | 2% | 0% | 1% |
| JPEG Q50 | 9% | **79%** | 3% | 1% | 0% | 8% |
| Resize 128 | 13% | 4% | **74%** | 7% | 0% | 1% |
| Resize 64 | 1% | 1% | 7% | **85%** | 5% | 2% |
| Resize 32 | 0% | 0% | 0% | 6% | **93%** | 1% |
| 128 + JPEG 50 | 0% | 5% | 2% | 2% | 0% | **91%** |

Router choice per condition (from degraded seen-generator validation): Clean: MLP, JPEG Q50: MLP/fusion, Resize 128: MLP, Resize 64: fusion, Resize 32: LR/fusion, 128 + JPEG 50: fusion.
