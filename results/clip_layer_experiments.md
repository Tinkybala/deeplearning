# Intermediate CLIP blocks, and CLIP + DINOv2, with nested selection

Single detectors with the clean seen-generator validation threshold. **[clean+both]** = fit on clean + augmented images (JPEG 30–95 and/or resize to 32/64/128 px, p = 0.5 each). **CLIP Lk** = CLS token of the residual stream after block k of OpenAI CLIP ViT-L/14. **fusion** = mean of the DINOv2 block-20 MLP's and the CLIP MLP's logits, each standardised on clean seen-val. **Inner** = nested leave-one-generator-out balanced accuracy, mean over the six standard conditions (never uses the outer test generator). Mean over 7 holdouts; **DINOv2 L20 MLP [clean+both]** is the final detector.

## All candidates, sorted by inner score

| Candidate | Inner (6 cond.) | Seen val | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion DINOv2 L20 + CLIP L16 [clean+both] | 81.7% | 98.7% | 93.0% | 81.6% | 90.1% | 81.4% | 72.4% | 76.4% | 79.6% | 82.5% |
| DINOv2 L12-24 LR [clean+both] | 81.3% | 96.0% | 89.3% | 83.8% | 87.5% | 82.3% | 73.8% | 76.7% | 86.1% | 82.2% |
| DINOv2 L20 + CLIP L16 MLP [clean+both] | 80.8% | 98.9% | 93.1% | 81.0% | 89.6% | 81.0% | 72.4% | 76.0% | 72.2% | 82.2% |
| DINOv2 L20 + CLIP L20 MLP [clean+both] | 80.6% | 96.8% | 92.2% | 83.6% | 87.9% | 81.1% | 72.2% | 76.5% | 80.8% | 82.2% |
| **DINOv2 L20 MLP [clean+both]** | 80.6% | 95.1% | 88.7% | 83.2% | 86.7% | 80.7% | 72.3% | 76.4% | 85.5% | 81.3% |
| fusion DINOv2 L20 + CLIP L24 [clean+both] | 80.5% | 96.6% | 91.2% | 85.8% | 87.4% | 81.6% | 72.5% | 77.2% | 86.3% | 82.6% |
| fusion DINOv2 L20 + CLIP L20 [clean+both] | 80.4% | 97.0% | 91.6% | 81.4% | 87.9% | 80.3% | 70.9% | 74.5% | 79.2% | 81.1% |
| fusion DINOv2 L20 + CLIP L12 [clean+both] | 79.3% | 99.0% | 92.7% | 78.6% | 89.1% | 80.8% | 71.4% | 74.3% | 78.6% | 81.1% |
| DINOv2 L20 + CLIP L12 MLP [clean+both] | 78.0% | 98.7% | 92.3% | 76.6% | 88.1% | 77.8% | 69.4% | 71.9% | 67.2% | 79.4% |
| fusion DINOv2 L20 + CLIP L8 [clean+both] | 76.0% | 98.1% | 89.4% | 71.9% | 85.1% | 78.0% | 69.6% | 69.6% | 75.4% | 77.3% |
| CLIP L20 MLP [clean+both] | 75.0% | 96.8% | 91.0% | 77.9% | 84.7% | 74.1% | 65.8% | 68.0% | 73.1% | 76.9% |
| CLIP L24 MLP [clean+both] | 75.0% | 93.6% | 88.4% | 80.0% | 81.3% | 73.0% | 64.7% | 69.2% | 79.6% | 76.1% |
| CLIP L16 LR [clean+both] | 74.6% | 97.4% | 90.5% | 74.4% | 83.5% | 72.2% | 65.0% | 66.1% | 69.5% | 75.3% |
| CLIP L20 LR [clean+both] | 73.9% | 94.7% | 88.8% | 75.9% | 82.0% | 71.5% | 62.9% | 66.7% | 72.2% | 74.6% |
| CLIP L16 MLP [clean+both] | 73.4% | 98.6% | 92.1% | 70.2% | 86.1% | 72.4% | 66.0% | 63.5% | 66.5% | 75.1% |
| CLIP L24 LR [clean+both] | 72.6% | 91.5% | 86.0% | 76.2% | 78.4% | 69.8% | 62.8% | 65.3% | 77.6% | 73.1% |
| CLIP L12 LR [clean+both] | 72.4% | 97.8% | 89.9% | 71.6% | 81.8% | 71.0% | 64.1% | 64.5% | 70.0% | 73.8% |
| CLIP L12 MLP [clean+both] | 71.6% | 98.6% | 90.6% | 66.3% | 85.5% | 72.0% | 64.7% | 63.1% | 68.2% | 73.7% |
| CLIP L8 LR [clean+both] | 67.4% | 95.2% | 81.2% | 64.4% | 71.2% | 65.8% | 63.6% | 64.4% | 59.2% | 68.4% |
| CLIP L8 MLP [clean+both] | 66.2% | 97.4% | 81.6% | 60.7% | 73.4% | 65.8% | 61.4% | 56.1% | 63.1% | 66.5% |
| CLIP L24 MLP [clean] | 65.8% | 94.1% | 87.8% | 68.0% | 67.7% | 58.3% | 51.1% | 63.5% | 74.6% | 66.1% |
| CLIP L20 MLP [clean] | 64.9% | 97.1% | 92.1% | 58.3% | 71.4% | 59.6% | 50.4% | 53.6% | 65.7% | 64.2% |
| CLIP L16 MLP [clean] | 62.9% | 99.0% | 92.1% | 51.4% | 69.3% | 60.8% | 51.0% | 51.1% | 59.1% | 62.6% |
| CLIP L12 MLP [clean] | 62.0% | 98.9% | 92.0% | 50.3% | 77.3% | 51.7% | 50.1% | 50.2% | 56.1% | 61.9% |

## Difference from the final detector (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0. Rows below the two rules are the top candidates by inner score.

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nested pick (mean of 6 conditions) − final | +2.5* [+1.9, +3.1] | -0.6 [-1.3, +0.1] | +2.1* [+1.5, +2.7] | -0.3 [-0.9, +0.4] | -0.4 [-1.2, +0.5] | -0.6 [-1.4, +0.2] | -3.0* [-3.7, -2.2] | +0.5* [+0.2, +0.8] |
| nested pick (clean) − final | +4.0* [+3.4, +4.6] | -2.0* [-2.6, -1.3] | +2.5* [+1.9, +3.1] | +0.1 [-0.6, +0.8] | -0.2 [-1.0, +0.6] | -0.1 [-0.9, +0.7] | -9.1* [-9.8, -8.4] | +0.7* [+0.4, +1.0] |
| fusion DINOv2 L20 + CLIP L16 [clean+both] − final | +4.3* [+3.7, +4.9] | -1.5* [-2.2, -0.9] | +3.4* [+2.8, +3.9] | +0.8* [+0.2, +1.3] | +0.1 [-0.6, +0.8] | +0.1 [-0.6, +0.8] | -6.0* [-6.6, -5.4] | +1.2* [+0.9, +1.4] |
| DINOv2 L12-24 LR [clean+both] − final | +0.6 [-0.1, +1.2] | +0.6 [-0.2, +1.4] | +0.8* [+0.1, +1.6] | +1.6* [+0.7, +2.5] | +1.5* [+0.5, +2.6] | +0.3 [-0.6, +1.4] | +0.5 [-0.2, +1.3] | +0.9* [+0.6, +1.3] |
| DINOv2 L20 + CLIP L16 MLP [clean+both] − final | +4.3* [+3.7, +5.0] | -2.2* [-2.9, -1.5] | +2.9* [+2.2, +3.5] | +0.3 [-0.4, +1.2] | +0.2 [-0.7, +1.1] | -0.4 [-1.2, +0.6] | -13.3* [-14.1, -12.5] | +0.9* [+0.5, +1.2] |
| DINOv2 L20 + CLIP L20 MLP [clean+both] − final | +3.5* [+2.9, +4.1] | +0.4 [-0.2, +1.1] | +1.2* [+0.5, +1.9] | +0.4 [-0.4, +1.2] | -0.1 [-1.0, +0.8] | +0.2 [-0.7, +1.0] | -4.8* [-5.4, -4.0] | +0.9* [+0.6, +1.2] |

## Picks per outer holdout

| Held-out generator | nested pick (mean of 6 conditions) | nested pick (clean) |
|---|---|---|
| BigGAN | fusion DINOv2 L20 + CLIP L16 [clean+both] | DINOv2 L20 + CLIP L12 MLP [clean+both] |
| VQDM | fusion DINOv2 L20 + CLIP L16 [clean+both] | fusion DINOv2 L20 + CLIP L16 [clean+both] |
| SD v1.5 | fusion DINOv2 L20 + CLIP L16 [clean+both] | fusion DINOv2 L20 + CLIP L16 [clean+both] |
| Wukong | DINOv2 L12-24 LR [clean+both] | DINOv2 L20 + CLIP L16 MLP [clean+both] |
| ADM | fusion DINOv2 L20 + CLIP L16 [clean+both] | fusion DINOv2 L20 + CLIP L16 [clean+both] |
| GLIDE | fusion DINOv2 L20 + CLIP L16 [clean+both] | DINOv2 L20 + CLIP L16 MLP [clean+both] |
| Midjourney | DINOv2 L12-24 LR [clean+both] | DINOv2 L20 + CLIP L16 MLP [clean+both] |

Nested picks (mean of 6 conditions): fusion DINOv2 L20 + CLIP L16 [clean+both] ×5, DINOv2 L12-24 LR [clean+both] ×2.

## Clean balanced accuracy per held-out generator

| Held-out generator | DINOv2 L20 MLP [clean+both] | fusion DINOv2 L20 + CLIP L16 [clean+both] | DINOv2 L12-24 LR [clean+both] | DINOv2 L20 + CLIP L16 MLP [clean+both] | DINOv2 L20 + CLIP L20 MLP [clean+both] |
|---|---:|---:|---:|---:|---:|
| BigGAN | 97.1% | **99.1%** | 95.7% | 97.8% | 97.7% |
| VQDM | 92.3% | 95.8% | 91.9% | **96.7%** | 96.5% |
| SD v1.5 | 92.1% | 97.5% | 94.0% | **98.4%** | 97.3% |
| Wukong | 92.4% | 96.1% | 93.4% | **97.4%** | 95.7% |
| ADM | 72.9% | 76.6% | **78.6%** | 77.5% | 77.7% |
| GLIDE | 96.1% | **98.5%** | 93.7% | 98.2% | 98.2% |
| Midjourney | 78.3% | **87.5%** | 78.0% | 85.5% | 82.6% |

## JPEG Q50 balanced accuracy per held-out generator

| Held-out generator | DINOv2 L20 MLP [clean+both] | fusion DINOv2 L20 + CLIP L16 [clean+both] | DINOv2 L12-24 LR [clean+both] | DINOv2 L20 + CLIP L16 MLP [clean+both] | DINOv2 L20 + CLIP L20 MLP [clean+both] |
|---|---:|---:|---:|---:|---:|
| BigGAN | 94.4% | 94.6% | 91.5% | 94.2% | **95.0%** |
| VQDM | 88.6% | 85.7% | **89.8%** | 87.4% | 86.7% |
| SD v1.5 | 86.8% | 82.5% | 88.3% | 79.7% | **89.4%** |
| Wukong | 86.8% | 76.5% | 87.8% | 80.0% | **88.3%** |
| ADM | 70.2% | 72.4% | **76.1%** | 71.2% | 68.2% |
| GLIDE | 90.8% | 95.6% | 93.4% | **96.5%** | 93.1% |
| Midjourney | **64.7%** | 64.2% | 59.5% | 58.1% | 64.3% |
