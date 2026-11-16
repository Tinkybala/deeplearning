# Size-matched control for the native-resolution gain

DINOv2 MLP probes (main runs' recipe) fit on clean seen-generator training images, one view each; leave-one-generator-out, clean seen-val threshold. **native** views keep each image's own size, which differs by class in GenImage. **448** views first resize every image, of both classes, in training and test, to shorter side 448 (bicubic): one shorter side and exactly 4 tiles for every image. **crop** = shorter side 256, centre 224; **centre** = central 224 px; **tile mean** = mean CLS over the 224 px tiles. **Format q96** = both classes re-saved as JPEG q96 at native size (before any resizing). Mean over the 7 holdouts.

## Balanced accuracy

| Block | View | Clean | Format q96 |
|---|---|---:|---:|
| 20 | downscaled crop (main runs) | 88.6% | 86.1% |
| 20 | native centre | 92.2% | 90.0% |
| 20 | native tile mean | 93.0% | 91.1% |
| 20 | 448 crop | 88.3% | 86.7% |
| 20 | 448 centre | 90.8% | 87.3% |
| 20 | 448 tile mean | 91.4% | 85.9% |
| 24 | downscaled crop (main runs) | 81.5% | 80.3% |
| 24 | native centre | 85.3% | 83.6% |
| 24 | native tile mean | 86.3% | 84.2% |
| 24 | 448 crop | 81.2% | 79.9% |
| 24 | 448 centre | 82.8% | 80.1% |
| 24 | 448 tile mean | 86.9% | 82.9% |

## AUROC

| Block | View | Clean | Format q96 |
|---|---|---:|---:|
| 20 | downscaled crop (main runs) | 0.956 | 0.944 |
| 20 | native centre | 0.981 | 0.975 |
| 20 | native tile mean | 0.984 | 0.981 |
| 20 | 448 crop | 0.958 | 0.950 |
| 20 | 448 centre | 0.974 | 0.956 |
| 20 | 448 tile mean | 0.981 | 0.965 |
| 24 | downscaled crop (main runs) | 0.899 | 0.889 |
| 24 | native centre | 0.930 | 0.920 |
| 24 | native tile mean | 0.942 | 0.930 |
| 24 | 448 crop | 0.900 | 0.890 |
| 24 | 448 centre | 0.914 | 0.892 |
| 24 | 448 tile mean | 0.949 | 0.926 |

## Difference from the downscaled crop, balanced accuracy (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Block | View − downscaled crop | Clean | Format q96 |
|---|---|---:|---:|
| 20 | native centre | +3.5* [+2.9, +4.3] | +3.9* [+3.1, +4.6] |
| 20 | native tile mean | +4.4* [+3.7, +5.1] | +5.0* [+4.3, +5.8] |
| 20 | 448 crop | -0.3 [-0.8, +0.1] | +0.6* [+0.2, +1.1] |
| 20 | 448 centre | +2.2* [+1.5, +2.9] | +1.2* [+0.5, +2.0] |
| 20 | 448 tile mean | +2.8* [+2.1, +3.6] | -0.2 [-1.0, +0.6] |
| 24 | native centre | +3.8* [+3.0, +4.5] | +3.4* [+2.6, +4.2] |
| 24 | native tile mean | +4.8* [+4.0, +5.7] | +3.9* [+3.1, +4.8] |
| 24 | 448 crop | -0.3 [-0.7, +0.2] | -0.4 [-0.8, +0.0] |
| 24 | 448 centre | +1.3* [+0.4, +2.1] | -0.1 [-1.0, +0.7] |
| 24 | 448 tile mean | +5.4* [+4.5, +6.3] | +2.7* [+1.7, +3.6] |

## Block 20, Clean, per held-out generator

| Held-out generator | downscaled crop (main runs) | native centre | native tile mean | 448 crop | 448 centre | 448 tile mean |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | 95.5% | 97.8% | **98.5%** | 94.8% | 95.9% | 96.8% |
| VQDM | 89.5% | 95.3% | **96.7%** | 90.5% | 91.3% | 89.1% |
| SD v1.5 | 92.8% | **97.9%** | 97.8% | 92.2% | 96.5% | 96.5% |
| Wukong | 91.9% | 96.2% | **97.1%** | 90.5% | 94.5% | 96.0% |
| ADM | 74.3% | 78.1% | 83.9% | 75.8% | 79.2% | **84.2%** |
| GLIDE | 94.6% | 97.9% | **98.3%** | 95.9% | 97.8% | **98.3%** |
| Midjourney | 81.8% | **82.0%** | 78.6% | 78.3% | 80.4% | 79.1% |

## Block 20, Format q96, per held-out generator

| Held-out generator | downscaled crop (main runs) | native centre | native tile mean | 448 crop | 448 centre | 448 tile mean |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | 92.9% | 95.7% | **96.6%** | 94.2% | 95.0% | 96.1% |
| VQDM | 87.0% | 93.7% | **95.8%** | 88.5% | 84.0% | 71.7% |
| SD v1.5 | 91.8% | **97.3%** | 96.6% | 91.5% | 95.8% | 95.2% |
| Wukong | 91.0% | 93.6% | 94.6% | 89.6% | 92.9% | **94.9%** |
| ADM | 67.6% | 71.1% | **78.6%** | 71.0% | 66.6% | 69.6% |
| GLIDE | 92.4% | 97.3% | **97.4%** | 94.8% | 96.6% | 95.7% |
| Midjourney | 79.9% | **81.0%** | 77.9% | 77.2% | 80.4% | 78.0% |

## Class recall, block 20 (real / generated)

| View | Clean | Format q96 |
|---|---:|---:|
| downscaled crop (main runs) | 94% / 84% | 94% / 78% |
| native centre | 97% / 87% | 97% / 83% |
| native tile mean | 97% / 89% | 97% / 85% |
| 448 crop | 94% / 83% | 94% / 79% |
| 448 centre | 96% / 86% | 96% / 79% |
| 448 tile mean | 97% / 86% | 97% / 75% |

## Real-photo recall by native size, block 20 (pooled over the 7 holdouts). 1 tile = both sides under 448 px, 2 tiles = one side at least 448 px, 4 tiles = both

| View | Clean, 1 tile (n = 156) | Clean, 2 tiles (n = 2802) | Clean, 4 tiles (n = 542) | Format q96, 1 tile (n = 156) | Format q96, 2 tiles (n = 2802) | Format q96, 4 tiles (n = 542) |
|---|---:|---:|---:|---:|---:|---:|
| downscaled crop (main runs) | 95.5% | 94.6% | 88.7% | 96.8% | 94.6% | 88.7% |
| native centre | 97.4% | 97.8% | 93.0% | 98.1% | 97.9% | 93.4% |
| native tile mean | 93.6% | 98.1% | 93.5% | 94.2% | 98.2% | 94.1% |
| 448 crop | 92.9% | 95.0% | 88.9% | 94.2% | 95.0% | 89.3% |
| 448 centre | 91.7% | 96.8% | 89.9% | 93.6% | 96.8% | 90.2% |
| 448 tile mean | 95.5% | 98.4% | 88.9% | 95.5% | 98.5% | 89.3% |
