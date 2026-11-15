# Backbone and layer comparison (frozen features)

Leave-one-generator-out over the 7 GenImage generators, same split, threshold rule and test conditions as the main runs. **LR** = logistic regression (C=0.1, class-balanced), **MLP** = the main runs' MLP probe (256 hidden, seed 42); both on standardised features. **Family out** = clean balanced accuracy with the held-out generator's whole family removed from training (SD v1.5 + Wukong, ADM + GLIDE), 7-generator mean as in A1. **Format 256+q96** = both classes resized to 256 and saved as JPEG q96 (E1). The router chooses, per condition, among all rows above it using only the seen generators' degraded validation images. **Seen val** = balanced accuracy on the clean seen-generator validation split (available without any test data, so it can be used to choose a layer). Mean over the 7 held-out generators; best per column in bold.

## Balanced accuracy

| Features | Probe | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) | Family out | AUROC clean | Seen val |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 clean+diff MLP (main runs) |  | 85.3% | 74.7% | 81.5% | 74.1% | 61.6% | 68.1% | 82.5% | 74.2% | 84.2% | 0.927 | 89.9% |
| DINOv2 L8 | LR | 75.4% | 56.7% | 52.2% | 50.4% | 50.3% | 50.8% | 66.1% | 56.0% | 69.2% | 0.820 | 87.2% |
| DINOv2 L8 | MLP | 78.1% | 59.1% | 51.5% | 52.0% | 50.0% | 51.1% | 64.2% | 57.0% | 69.7% | 0.858 | 89.9% |
| DINOv2 L12 | LR | 82.7% | 54.0% | 63.3% | 52.9% | 50.1% | 50.3% | 70.7% | 58.9% | 77.3% | 0.920 | 95.0% |
| DINOv2 L12 | MLP | 83.6% | 53.1% | 65.4% | 52.8% | 50.0% | 50.0% | 67.5% | 59.2% | 76.4% | 0.937 | 97.0% |
| DINOv2 L16 | LR | 86.5% | 62.3% | 68.5% | 54.0% | 49.8% | 53.8% | 80.3% | 62.5% | 82.0% | 0.946 | 95.8% |
| DINOv2 L16 | MLP | 85.3% | 56.6% | 78.6% | 55.9% | 50.9% | 50.7% | 76.9% | 63.0% | 78.1% | 0.947 | 96.7% |
| DINOv2 L20 | LR | 87.6% | 74.1% | 85.2% | 77.1% | 64.3% | 66.0% | **84.1%** | 75.7% | 86.0% | 0.948 | 94.0% |
| DINOv2 L20 | MLP | 88.6% | 74.7% | 85.8% | 78.5% | **66.4%** | 66.3% | 83.4% | 76.7% | 84.9% | 0.956 | 94.6% |
| DINOv2 L24 | LR | 80.5% | 74.8% | 77.9% | 73.1% | 63.8% | 70.1% | 78.9% | 73.4% | 79.3% | 0.883 | 84.8% |
| DINOv2 L24 | MLP | 81.5% | 76.9% | 79.1% | 74.4% | 66.1% | 72.4% | 80.3% | 75.1% | 80.1% | 0.899 | 85.5% |
| DINOv2 L16 patch | LR | 85.0% | 72.1% | 82.6% | 75.0% | 55.8% | 60.8% | 82.1% | 71.9% | 81.5% | 0.933 | 93.8% |
| DINOv2 L16 patch | MLP | 86.6% | 76.0% | 84.2% | 77.4% | 60.5% | 64.4% | 84.0% | 74.9% | 81.1% | 0.947 | 95.6% |
| DINOv2 L24 patch | LR | 83.0% | 79.1% | 80.8% | 75.8% | 63.6% | 73.7% | 82.2% | 76.0% | 81.4% | 0.912 | 88.2% |
| DINOv2 L24 patch | MLP | 83.1% | 79.7% | 81.2% | 75.7% | 62.9% | **74.7%** | 82.2% | 76.2% | 81.1% | 0.915 | 87.7% |
| DINOv2 L12-24 | LR | 88.5% | 66.7% | 84.7% | 72.0% | 53.3% | 56.0% | 83.2% | 70.2% | 85.5% | 0.970 | 97.3% |
| DINOv2 L12-24 | MLP | 88.3% | 60.3% | 84.3% | 66.5% | 51.1% | 52.3% | 82.0% | 67.1% | 82.3% | 0.955 | 97.1% |
| CLIP | LR | 86.6% | 59.5% | 65.2% | 58.2% | 51.2% | 53.7% | 73.4% | 62.4% | 84.2% | 0.936 | 92.0% |
| CLIP | MLP | 87.7% | 63.9% | 66.9% | 57.4% | 51.1% | 58.6% | 74.6% | 64.3% | 85.9% | 0.945 | 93.4% |
| DINOv2 L24 + CLIP | LR | 87.6% | 67.0% | 75.9% | 68.8% | 55.4% | 59.8% | 79.1% | 69.1% | 86.6% | 0.948 | 92.8% |
| DINOv2 L24 + CLIP | MLP | 88.8% | 74.1% | 80.6% | 71.0% | 58.0% | 68.5% | 81.8% | 73.5% | 87.6% | 0.956 | 94.0% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP |  | **90.0%** | 69.8% | 77.8% | 66.8% | 54.5% | 66.2% | 81.1% | 70.9% | 89.0% | 0.964 | 94.8% |
| router |  | 87.3% | **82.4%** | **87.1%** | **79.4%** | 65.2% | 73.8% | 83.4% | 79.2% |  |  |  |

## Difference from the reference (DINOv2 clean+diff MLP (main runs)), balanced accuracy (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Method − reference | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 L20 LR | +2.3* [+1.4, +3.2] | -0.7 [-1.7, +0.4] | +3.7* [+2.6, +4.7] | +3.0* [+1.8, +4.4] | +2.8* [+1.2, +4.2] | -2.1* [-3.4, -0.9] | +1.6* [+0.7, +2.6] |
| DINOv2 L20 MLP | +3.3* [+2.4, +4.2] | -0.0 [-1.1, +1.0] | +4.3* [+3.3, +5.4] | +4.4* [+3.2, +5.6] | +4.8* [+3.4, +6.2] | -1.7* [-2.9, -0.5] | +0.9 [-0.0, +1.8] |
| DINOv2 L16 MLP | -0.0 [-1.0, +1.0] | -18.2* [-19.2, -17.2] | -2.9* [-4.1, -1.7] | -18.2* [-19.3, -17.1] | -10.7* [-11.9, -9.5] | -17.4* [-18.4, -16.3] | -5.6* [-6.6, -4.5] |
| DINOv2 L12-24 LR | +3.1* [+2.3, +4.0] | -8.1* [-9.1, -7.0] | +3.1* [+2.2, +4.2] | -2.1* [-3.2, -1.0] | -8.2* [-9.4, -7.1] | -12.0* [-13.1, -10.9] | +0.7 [-0.2, +1.6] |
| DINOv2 L24 patch MLP | -2.3* [-3.2, -1.3] | +4.9* [+3.9, +6.0] | -0.4 [-1.4, +0.7] | +1.6* [+0.4, +2.8] | +1.4* [+0.2, +2.6] | +6.6* [+5.4, +7.8] | -0.3 [-1.3, +0.7] |
| CLIP LR | +1.3* [+0.2, +2.3] | -15.2* [-16.2, -14.2] | -16.4* [-17.5, -15.2] | -15.9* [-17.1, -14.7] | -10.3* [-11.5, -9.3] | -14.4* [-15.5, -13.3] | -9.1* [-10.2, -8.0] |
| CLIP MLP | +2.4* [+1.4, +3.4] | -10.8* [-11.9, -9.8] | -14.6* [-15.8, -13.4] | -16.7* [-17.9, -15.6] | -10.4* [-11.6, -9.3] | -9.5* [-10.7, -8.2] | -7.9* [-9.0, -6.9] |
| DINOv2 L24 + CLIP MLP | +3.5* [+2.7, +4.3] | -0.7 [-1.6, +0.3] | -0.9 [-1.9, +0.0] | -3.1* [-4.2, -2.0] | -3.6* [-4.7, -2.5] | +0.4 [-0.7, +1.5] | -0.7 [-1.5, +0.1] |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | +4.7* [+4.0, +5.4] | -4.9* [-5.8, -4.0] | -3.7* [-4.7, -2.8] | -7.3* [-8.3, -6.2] | -7.1* [-8.2, -6.1] | -1.8* [-2.8, -0.8] | -1.4* [-2.3, -0.7] |
| router | +2.0* [+1.1, +2.9] | +7.6* [+6.5, +8.7] | +5.6* [+4.6, +6.6] | +5.3* [+4.2, +6.5] | +3.6* [+2.3, +4.9] | +5.7* [+4.6, +6.9] | +0.9 [-0.0, +1.9] |

## Clean balanced accuracy per held-out generator

| Held-out generator | DINOv2 clean+diff MLP (main runs) | DINOv2 L24 LR | DINOv2 L16 MLP | DINOv2 L20 MLP | CLIP MLP | DINOv2 L24 + CLIP MLP | fusion: DINOv2 clean+diff MLP + CLIP MLP | router |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BigGAN | 92.8% | 91.5% | 96.3% | 95.5% | 93.9% | 95.1% | 97.1% | **98.2%** |
| VQDM | 86.1% | 85.4% | 69.3% | 89.5% | 89.4% | **93.5%** | 92.3% | 88.5% |
| SD v1.5 | 85.6% | 79.6% | 94.9% | 92.8% | 90.8% | 91.9% | 92.3% | **95.3%** |
| Wukong | 86.5% | 81.2% | **92.3%** | 91.9% | 88.2% | 89.1% | 90.3% | 92.2% |
| ADM | 76.6% | 69.8% | 67.0% | 74.3% | 76.3% | 75.0% | **78.4%** | 67.1% |
| GLIDE | 91.5% | 87.5% | 95.2% | 94.6% | 95.2% | 96.7% | **96.9%** | 94.9% |
| Midjourney | 78.1% | 68.6% | 82.0% | 81.8% | 80.3% | 80.4% | **82.7%** | 75.0% |

## Router choice (on degraded seen-generator validation)

| Held-out generator | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 |
|---|---|---|---|---|---|---|---|
| BigGAN | DINOv2 L12-24 MLP | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | DINOv2 L16 patch MLP |
| VQDM | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 MLP | DINOv2 L20 MLP | DINOv2 L16 MLP |
| SD v1.5 | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L12-24 MLP |
| Wukong | DINOv2 L12 MLP | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | DINOv2 L16 patch MLP |
| ADM | DINOv2 L12 MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L24 patch MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP |
| GLIDE | DINOv2 L12 MLP | DINOv2 L16 patch MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | DINOv2 L16 patch MLP |
| Midjourney | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | DINOv2 L12-24 LR |
