# Held-out degradations: robustness to transformations never seen in training

Training augmentation (the *aug.* detectors) used JPEG quality 30-95 and downscaling to 32, 64 or 128 px; the four **seen-type** conditions use the same operations. The six **held-out** conditions are outside that training distribution: JPEG Q20 (below the trained range), WebP Q50 (another codec), Gaussian blur (radius 1 px, native resolution), additive Gaussian noise (sigma 5/255), and downscaling to 48 or 96 px (sizes between the trained ones). Leave-one-generator-out, clean seen-val threshold, mean over the 7 holdouts.

## Balanced accuracy

| Detector | Clean | JPEG Q50 | Resize 64 | Resize 32 | JPEG Q20 | WebP Q50 | Blur r1 | Noise 5/255 | Resize 48 | Resize 96 | Mean, held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final layer (block 24) MLP, clean | 81.5% | 76.9% | 74.4% | 66.1% | 74.0% | 77.0% | 80.4% | 78.3% | 71.6% | 77.3% | 76.4% |
| Block-20 MLP, clean | 88.6% | 74.7% | 78.5% | 66.4% | 67.2% | 80.9% | 87.3% | 79.6% | 74.1% | 83.7% | 78.8% |
| Final detector (block-20 MLP, clean + aug.) | 88.7% | 83.2% | 80.7% | 72.3% | 80.8% | 78.2% | 87.8% | 80.9% | 77.2% | 84.9% | 81.6% |
| Fusion with DINOv3 block 24 (clean + aug.) | 91.7% | 87.0% | 84.2% | 74.8% | 83.8% | 80.7% | 90.6% | 86.0% | 80.7% | 87.7% | 84.9% |

## AUROC

| Detector | Clean | JPEG Q50 | Resize 64 | Resize 32 | JPEG Q20 | WebP Q50 | Blur r1 | Noise 5/255 | Resize 48 | Resize 96 | Mean, held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final layer (block 24) MLP, clean | 0.899 | 0.852 | 0.834 | 0.729 | 0.828 | 0.859 | 0.889 | 0.874 | 0.800 | 0.866 | 0.852 |
| Block-20 MLP, clean | 0.956 | 0.915 | 0.870 | 0.732 | 0.883 | 0.903 | 0.947 | 0.931 | 0.823 | 0.916 | 0.901 |
| Final detector (block-20 MLP, clean + aug.) | 0.959 | 0.930 | 0.892 | 0.799 | 0.888 | 0.902 | 0.953 | 0.942 | 0.860 | 0.928 | 0.912 |
| Fusion with DINOv3 block 24 (clean + aug.) | 0.976 | 0.957 | 0.924 | 0.830 | 0.931 | 0.941 | 0.973 | 0.962 | 0.893 | 0.952 | 0.942 |

## Differences, balanced accuracy (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Comparison | Clean | JPEG Q50 | Resize 64 | Resize 32 | JPEG Q20 | WebP Q50 | Blur r1 | Noise 5/255 | Resize 48 | Resize 96 | Mean, held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final detector (block-20 MLP, clean + aug.) − Final layer (block 24) MLP, clean | +7.2* [+6.3, +8.2] | +6.3* [+5.3, +7.4] | +6.3* [+5.1, +7.5] | +6.2* [+4.9, +7.5] | +6.8* [+5.5, +7.9] | +1.3* [+0.1, +2.4] | +7.4* [+6.4, +8.4] | +2.7* [+1.7, +3.7] | +5.6* [+4.3, +6.8] | +7.6* [+6.6, +8.7] | +5.2* [+4.8, +5.7] |
| Final detector (block-20 MLP, clean + aug.) − Block-20 MLP, clean | +0.1 [-0.4, +0.6] | +8.5* [+7.8, +9.2] | +2.2* [+1.4, +3.0] | +5.9* [+4.8, +6.9] | +13.6* [+12.6, +14.5] | -2.7* [-3.3, -2.0] | +0.5 [-0.1, +1.1] | +1.3* [+0.8, +1.9] | +3.1* [+2.2, +4.0] | +1.2* [+0.5, +1.9] | +2.8* [+2.5, +3.1] |
| Fusion with DINOv3 block 24 (clean + aug.) − Final detector (block-20 MLP, clean + aug.) | +2.9* [+2.4, +3.5] | +3.8* [+3.1, +4.4] | +3.6* [+2.8, +4.3] | +2.6* [+1.7, +3.5] | +3.0* [+2.2, +3.8] | +2.4* [+1.7, +3.2] | +2.8* [+2.1, +3.4] | +5.1* [+4.4, +5.8] | +3.5* [+2.7, +4.3] | +2.8* [+2.2, +3.5] | +3.3* [+3.0, +3.6] |

## Class recall (real / generated)

| Detector | Clean | JPEG Q50 | Resize 64 | Resize 32 | JPEG Q20 | WebP Q50 | Blur r1 | Noise 5/255 | Resize 48 | Resize 96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final layer (block 24) MLP, clean | 86% / 77% | 82% / 72% | 71% / 77% | 54% / 78% | 83% / 65% | 77% / 77% | 83% / 78% | 90% / 67% | 66% / 77% | 77% / 77% |
| Block-20 MLP, clean | 94% / 84% | 97% / 52% | 74% / 83% | 63% / 70% | 98% / 36% | 75% / 87% | 91% / 84% | 97% / 63% | 73% / 76% | 82% / 86% |
| Final detector (block-20 MLP, clean + aug.) | 94% / 83% | 95% / 72% | 84% / 77% | 74% / 70% | 88% / 74% | 65% / 91% | 93% / 83% | 98% / 64% | 83% / 71% | 87% / 82% |
| Fusion with DINOv3 block 24 (clean + aug.) | 96% / 88% | 95% / 79% | 87% / 82% | 72% / 78% | 94% / 73% | 68% / 93% | 95% / 86% | 98% / 74% | 84% / 78% | 91% / 85% |

## Final detector per held-out generator, held-out conditions

| Held-out generator | JPEG Q20 | WebP Q50 | Blur r1 | Noise 5/255 | Resize 48 | Resize 96 |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | 91.8% | 83.6% | 95.9% | 95.6% | 83.8% | 92.7% |
| VQDM | 87.0% | 79.6% | 90.0% | 86.6% | 78.1% | 85.7% |
| SD v1.5 | 82.7% | 76.7% | 91.8% | 86.7% | 82.5% | 88.0% |
| Wukong | 82.3% | 78.7% | 91.3% | 85.3% | 83.9% | 90.3% |
| ADM | 75.1% | 83.5% | 76.8% | 59.8% | 64.1% | 71.4% |
| GLIDE | 86.5% | 83.2% | 91.3% | 80.0% | 77.8% | 88.1% |
| Midjourney | 60.1% | 62.4% | 77.5% | 72.6% | 70.2% | 77.9% |
