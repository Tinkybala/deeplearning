# Intermediate DINOv3 blocks, and DINOv3 + DINOv2, with nested selection

Single detectors with the clean seen-generator validation threshold. **[clean+both]** = fit on clean + augmented images (JPEG 30–95 and/or resize to 32/64/128 px, p = 0.5 each). **DINOv3 Lk** = CLS token of the residual stream after block k of DINOv3 ViT-L/16 (LVD-1689M, through its final norm). **fusion** = mean of the DINOv2 block-20 MLP's and the DINOv3 MLP's logits, each standardised on clean seen-val. **Inner** = nested leave-one-generator-out balanced accuracy, mean over the six standard conditions (never uses the outer test generator). Mean over 7 holdouts; **DINOv2 L20 MLP [clean+both]** is the final detector.

## All candidates, sorted by inner score

| Candidate | Inner (6 cond.) | Seen val | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion DINOv2 L20 + DINOv3 L24 [clean+both] | 83.8% | 96.4% | 91.7% | 87.0% | 88.7% | 84.2% | 74.8% | 79.9% | 88.6% | 84.4% |
| DINOv2 L20 + DINOv3 L20 MLP [clean+both] | 83.1% | 97.5% | 92.8% | 86.8% | 89.5% | 83.3% | 72.8% | 78.5% | 88.9% | 84.0% |
| fusion DINOv2 L20 + DINOv3 L20 [clean+both] | 82.5% | 97.3% | 92.4% | 85.6% | 89.5% | 83.5% | 73.4% | 78.7% | 88.3% | 83.8% |
| DINOv3 L20 LR [clean+both] | 82.5% | 97.1% | 93.9% | 87.4% | 89.9% | 82.8% | 71.3% | 76.8% | 90.0% | 83.7% |
| DINOv3 L20 MLP [clean+both] | 81.9% | 97.6% | 93.3% | 85.7% | 89.2% | 82.4% | 70.9% | 75.8% | 88.3% | 82.9% |
| DINOv2 L20 + DINOv3 L16 MLP [clean+both] | 81.6% | 98.0% | 92.7% | 83.6% | 89.5% | 82.3% | 72.3% | 76.8% | 83.5% | 82.9% |
| fusion DINOv2 L20 + DINOv3 L16 [clean+both] | 81.6% | 98.1% | 91.9% | 82.8% | 88.9% | 82.0% | 73.8% | 76.2% | 84.6% | 82.6% |
| DINOv2 L12-24 LR [clean+both] | 81.3% | 96.0% | 89.3% | 83.8% | 87.5% | 82.3% | 73.8% | 76.7% | 86.1% | 82.2% |
| fusion DINOv2 L20 + DINOv3 L8 [clean+both] | 81.1% | 97.9% | 91.4% | 80.6% | 89.0% | 82.6% | 76.7% | 82.4% | 83.1% | 83.8% |
| DINOv3 L24 MLP [clean+both] | 80.8% | 93.6% | 89.9% | 84.1% | 85.1% | 81.0% | 71.3% | 75.9% | 87.6% | 81.2% |
| **DINOv2 L20 MLP [clean+both]** | 80.6% | 95.1% | 88.7% | 83.2% | 86.7% | 80.7% | 72.3% | 76.4% | 85.5% | 81.3% |
| fusion DINOv2 L20 + DINOv3 L12 [clean+both] | 80.3% | 98.3% | 92.1% | 78.6% | 88.7% | 81.7% | 73.4% | 76.1% | 84.6% | 81.8% |
| DINOv2 L20 + DINOv3 L12 MLP [clean+both] | 79.3% | 97.7% | 90.8% | 80.8% | 86.7% | 79.4% | 71.4% | 74.6% | 79.8% | 80.6% |
| DINOv3 L16 MLP [clean+both] | 79.1% | 98.3% | 92.0% | 77.5% | 87.5% | 78.3% | 69.3% | 67.9% | 80.4% | 78.8% |
| DINOv3 L16 LR [clean+both] | 78.9% | 97.5% | 91.2% | 77.2% | 86.5% | 78.4% | 68.8% | 73.0% | 82.0% | 79.2% |
| DINOv3 L24 LR [clean+both] | 78.5% | 93.2% | 89.4% | 82.3% | 83.5% | 78.9% | 68.8% | 71.3% | 86.6% | 79.0% |
| DINOv3 L24 MLP [clean] | 77.8% | 93.8% | 89.7% | 79.8% | 84.2% | 80.0% | 68.0% | 69.3% | 87.0% | 78.5% |
| DINOv3 L12 LR [clean+both] | 74.5% | 97.0% | 89.4% | 68.6% | 82.3% | 72.3% | 66.9% | 73.1% | 75.4% | 75.4% |
| DINOv3 L8 LR [clean+both] | 74.5% | 93.8% | 83.9% | 73.0% | 76.5% | 72.8% | 70.5% | 76.9% | 70.8% | 75.6% |
| DINOv3 L8 MLP [clean+both] | 73.9% | 96.3% | 83.8% | 73.5% | 78.9% | 71.6% | 71.0% | 76.6% | 71.1% | 75.9% |
| DINOv3 L20 MLP [clean] | 73.5% | 98.0% | 92.9% | 72.3% | 87.7% | 73.3% | 55.2% | 61.5% | 88.0% | 73.8% |
| DINOv3 L12 MLP [clean+both] | 73.4% | 97.9% | 90.0% | 68.9% | 84.2% | 73.3% | 65.2% | 64.9% | 78.2% | 74.4% |
| DINOv3 L16 MLP [clean] | 66.8% | 98.5% | 91.3% | 59.3% | 85.2% | 60.6% | 51.3% | 53.1% | 81.5% | 66.8% |
| DINOv3 L12 MLP [clean] | 63.3% | 98.2% | 89.3% | 54.6% | 78.4% | 52.6% | 49.9% | 53.0% | 77.0% | 63.0% |

## Difference from the final detector (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0. Rows below the two rules are the top candidates by inner score.

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nested pick (mean of 6 conditions) − final | +2.9* [+2.4, +3.5] | +3.8* [+3.1, +4.5] | +2.0* [+1.3, +2.7] | +3.6* [+2.7, +4.3] | +2.6* [+1.7, +3.5] | +3.6* [+2.7, +4.4] | +3.1* [+2.5, +3.7] | +3.1* [+2.8, +3.4] |
| nested pick (clean) − final | +5.2* [+4.5, +5.9] | +3.6* [+2.8, +4.4] | +2.9* [+2.1, +3.7] | +1.7* [+0.7, +2.8] | -1.4* [-2.6, -0.2] | -0.3 [-1.4, +0.7] | +4.4* [+3.6, +5.2] | +1.9* [+1.6, +2.3] |
| fusion DINOv2 L20 + DINOv3 L24 [clean+both] − final | +2.9* [+2.4, +3.5] | +3.8* [+3.1, +4.5] | +2.0* [+1.3, +2.7] | +3.6* [+2.7, +4.3] | +2.6* [+1.7, +3.5] | +3.6* [+2.7, +4.4] | +3.1* [+2.5, +3.7] | +3.1* [+2.8, +3.4] |
| DINOv2 L20 + DINOv3 L20 MLP [clean+both] − final | +4.1* [+3.5, +4.6] | +3.7* [+3.0, +4.3] | +2.8* [+2.2, +3.4] | +2.6* [+1.9, +3.4] | +0.5 [-0.3, +1.4] | +2.1* [+1.4, +3.0] | +3.3* [+2.7, +4.0] | +2.6* [+2.3, +2.9] |
| fusion DINOv2 L20 + DINOv3 L20 [clean+both] − final | +3.6* [+3.2, +4.2] | +2.4* [+1.9, +3.0] | +2.8* [+2.2, +3.3] | +2.8* [+2.2, +3.5] | +1.2* [+0.4, +2.0] | +2.3* [+1.6, +3.1] | +2.7* [+2.2, +3.3] | +2.5* [+2.3, +2.8] |
| DINOv3 L20 LR [clean+both] − final | +5.2* [+4.5, +5.9] | +4.2* [+3.4, +5.1] | +3.2* [+2.3, +4.0] | +2.1* [+1.1, +3.1] | -1.0 [-2.2, +0.2] | +0.4 [-0.7, +1.5] | +4.5* [+3.7, +5.3] | +2.4* [+2.0, +2.7] |

## Picks per outer holdout

| Held-out generator | nested pick (mean of 6 conditions) | nested pick (clean) |
|---|---|---|
| BigGAN | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv3 L20 MLP [clean+both] |
| VQDM | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv3 L20 MLP [clean+both] |
| SD v1.5 | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv3 L20 LR [clean+both] |
| Wukong | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv3 L20 LR [clean+both] |
| ADM | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv3 L20 LR [clean+both] |
| GLIDE | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv3 L20 LR [clean+both] |
| Midjourney | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv3 L20 LR [clean+both] |

Nested picks (mean of 6 conditions): fusion DINOv2 L20 + DINOv3 L24 [clean+both] ×7.

## Clean balanced accuracy per held-out generator

| Held-out generator | DINOv2 L20 MLP [clean+both] | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv2 L20 + DINOv3 L20 MLP [clean+both] | fusion DINOv2 L20 + DINOv3 L20 [clean+both] | DINOv3 L20 LR [clean+both] |
|---|---:|---:|---:|---:|---:|
| BigGAN | 97.1% | 97.8% | 97.5% | **98.4%** | 97.2% |
| VQDM | 92.3% | **96.7%** | 95.8% | 95.1% | 95.0% |
| SD v1.5 | 92.1% | 95.7% | 96.4% | 96.1% | **96.7%** |
| Wukong | 92.4% | 95.4% | 96.5% | 96.0% | **97.1%** |
| ADM | 72.9% | 77.3% | 80.3% | 77.0% | **83.9%** |
| GLIDE | 96.1% | 95.3% | **96.4%** | **96.4%** | **96.4%** |
| Midjourney | 78.3% | 83.5% | 86.7% | 87.7% | **91.3%** |

## JPEG Q50 balanced accuracy per held-out generator

| Held-out generator | DINOv2 L20 MLP [clean+both] | fusion DINOv2 L20 + DINOv3 L24 [clean+both] | DINOv2 L20 + DINOv3 L20 MLP [clean+both] | fusion DINOv2 L20 + DINOv3 L20 [clean+both] | DINOv3 L20 LR [clean+both] |
|---|---:|---:|---:|---:|---:|
| BigGAN | 94.4% | **96.5%** | 95.4% | 95.4% | 92.6% |
| VQDM | 88.6% | 93.6% | **93.8%** | 89.8% | 92.5% |
| SD v1.5 | 86.8% | **91.8%** | 90.2% | 88.5% | 88.1% |
| Wukong | 86.8% | 90.2% | 87.7% | 89.3% | **90.4%** |
| ADM | 70.2% | 73.2% | 79.5% | 72.0% | **80.2%** |
| GLIDE | 90.8% | 93.9% | **94.8%** | 93.9% | 93.2% |
| Midjourney | 64.7% | 69.6% | 66.5% | 70.4% | **75.0%** |
