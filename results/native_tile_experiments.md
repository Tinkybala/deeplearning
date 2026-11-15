# Native resolution: downscaled crop vs native centre crop vs native tile mean

DINOv2 MLP probes (main runs' recipe) fit on clean seen-generator training images, one view each; leave-one-generator-out, clean seen-val threshold. **crop** = shorter side 256, centre 224 (main runs); **center** = central 224 px at native resolution; **tilemean** = mean CLS over up to four native-resolution 224 px tiles; **crop + tilemean** = the two MLPs' standardised logits averaged. **Format q96** = both classes re-saved as JPEG q96 at native size. Mean over the 7 holdouts.

## Balanced accuracy

| Block | View | Clean | JPEG Q50 | Format q96 |
|---|---|---:|---:|---:|
| 20 | downscaled crop (main runs) | 88.6% | 74.7% | 86.1% |
| 20 | native centre crop | 92.2% | 70.5% | 90.0% |
| 20 | native tile mean | 93.0% | 72.1% | 91.1% |
| 20 | crop + tilemean | 92.5% | 72.4% | 90.3% |
| 24 | downscaled crop (main runs) | 81.5% | 76.9% | 80.3% |
| 24 | native centre crop | 85.3% | 78.4% | 83.6% |
| 24 | native tile mean | 86.3% | 79.2% | 84.2% |
| 24 | crop + tilemean | 85.1% | 79.3% | 83.3% |

## AUROC

| Block | View | Clean | JPEG Q50 | Format q96 |
|---|---|---:|---:|---:|
| 20 | downscaled crop (main runs) | 0.956 | 0.915 | 0.944 |
| 20 | native centre crop | 0.981 | 0.908 | 0.975 |
| 20 | native tile mean | 0.984 | 0.919 | 0.981 |
| 20 | crop + tilemean | 0.981 | 0.933 | 0.975 |
| 24 | downscaled crop (main runs) | 0.899 | 0.852 | 0.889 |
| 24 | native centre crop | 0.930 | 0.868 | 0.920 |
| 24 | native tile mean | 0.942 | 0.876 | 0.930 |
| 24 | crop + tilemean | 0.935 | 0.878 | 0.924 |

## Difference from the downscaled crop, balanced accuracy (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Block | View − crop | Clean | JPEG Q50 | Format q96 |
|---|---|---:|---:|---:|
| 20 | native centre crop | +3.5* [+2.8, +4.2] | -4.2* [-5.0, -3.4] | +3.9* [+3.1, +4.6] |
| 20 | native tile mean | +4.4* [+3.7, +5.1] | -2.6* [-3.4, -1.8] | +5.0* [+4.2, +5.8] |
| 20 | crop + tilemean | +3.9* [+3.3, +4.4] | -2.3* [-2.9, -1.8] | +4.2* [+3.6, +4.7] |
| 24 | native centre crop | +3.8* [+2.9, +4.6] | +1.5* [+0.6, +2.3] | +3.4* [+2.6, +4.2] |
| 24 | native tile mean | +4.8* [+4.0, +5.7] | +2.3* [+1.4, +3.2] | +3.9* [+3.1, +4.8] |
| 24 | crop + tilemean | +3.6* [+3.0, +4.2] | +2.4* [+1.8, +3.0] | +3.0* [+2.4, +3.6] |

## Block 20, Clean, per held-out generator

| Held-out generator | downscaled crop (main runs) | native centre crop | native tile mean | crop + tilemean |
|---|---:|---:|---:|---:|
| BigGAN | 95.5% | 97.8% | 98.5% | **98.6%** |
| VQDM | 89.5% | 95.3% | **96.7%** | 94.7% |
| SD v1.5 | 92.8% | **97.9%** | 97.8% | 97.5% |
| Wukong | 91.9% | 96.2% | **97.1%** | 94.7% |
| ADM | 74.3% | 78.1% | **83.9%** | 79.7% |
| GLIDE | 94.6% | 97.9% | **98.3%** | 97.9% |
| Midjourney | 81.8% | 82.0% | 78.6% | **84.3%** |

## Block 20, JPEG Q50, per held-out generator

| Held-out generator | downscaled crop (main runs) | native centre crop | native tile mean | crop + tilemean |
|---|---:|---:|---:|---:|
| BigGAN | 86.9% | **87.8%** | 87.7% | 87.6% |
| VQDM | **77.8%** | 70.8% | 75.8% | 76.7% |
| SD v1.5 | **79.8%** | 77.3% | 73.3% | 76.8% |
| Wukong | **79.6%** | 74.0% | 73.6% | 74.2% |
| ADM | 57.5% | 55.1% | **61.2%** | 59.4% |
| GLIDE | **80.4%** | 73.5% | 80.3% | 75.9% |
| Midjourney | **60.9%** | 55.2% | 52.6% | 56.0% |

## Block 20, Format q96, per held-out generator

| Held-out generator | downscaled crop (main runs) | native centre crop | native tile mean | crop + tilemean |
|---|---:|---:|---:|---:|
| BigGAN | 92.9% | 95.7% | **96.6%** | 95.9% |
| VQDM | 87.0% | 93.7% | **95.8%** | 93.3% |
| SD v1.5 | 91.8% | **97.3%** | 96.6% | 96.4% |
| Wukong | 91.0% | 93.6% | **94.6%** | 92.9% |
| ADM | 67.6% | 71.1% | **78.6%** | 73.3% |
| GLIDE | 92.4% | 97.3% | **97.4%** | 96.7% |
| Midjourney | 79.9% | 81.0% | 77.9% | **83.3%** |

## Class recall, block 20 (real / generated)

| Block | View | Clean | JPEG Q50 | Format q96 |
|---|---|---:|---:|---:|
| 20 | downscaled crop (main runs) | 94% / 84% | 97% / 52% | 94% / 78% |
| 20 | native centre crop | 97% / 87% | 98% / 43% | 97% / 83% |
| 20 | native tile mean | 97% / 89% | 99% / 45% | 97% / 85% |
| 20 | crop + tilemean | 97% / 88% | 99% / 46% | 97% / 83% |
