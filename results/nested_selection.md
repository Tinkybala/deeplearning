# Nested leave-one-generator-out model selection

Candidates: the 22 frozen-feature probes of backbone_comparison, the main runs' DINOv2 clean+diff MLP (reference) and its fusion with the CLIP MLP. **Inner score** = balanced accuracy on a seen generator held out from a refit on the other five, averaged over the six inner folds; the outer held-out generator is never used for selection. **seen-val pick** = best clean seen-generator validation BA; **nested clean pick** = best inner clean BA; **nested one detector** = best inner BA averaged over the six standard conditions (one probe and its clean seen-val threshold for every condition); **router** = per condition, best on the seen generators' degraded validation images (as before); **nested router** = per condition, best inner BA under that condition, with the threshold re-picked on degraded seen-val. *DINOv2 L20 MLP* and *fusion* are the post-hoc best rows of backbone_comparison, shown for reference. Mean over the 7 held-out generators; best per column in bold.

## Balanced accuracy

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 clean+diff MLP (main runs) | 85.3% | 74.7% | 81.5% | 74.1% | 61.6% | 68.1% | 82.5% | 74.2% |
| DINOv2 L20 MLP | 88.6% | 74.7% | 85.8% | 78.5% | **66.4%** | 66.3% | 83.4% | 76.7% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | **90.0%** | 69.8% | 77.8% | 66.8% | 54.5% | 66.2% | 81.1% | 70.9% |
| seen-val pick | 87.7% | 62.2% | 80.8% | 67.0% | 52.6% | 53.3% | 81.2% | 67.3% |
| nested clean pick | 87.6% | 67.4% | 79.9% | 68.8% | 53.9% | 61.6% | 79.8% | 69.9% |
| nested one detector | 83.3% | 76.9% | 80.7% | 75.0% | 63.6% | 71.1% | 82.2% | 75.1% |
| router | 87.3% | 82.4% | **87.1%** | **79.4%** | 65.2% | **73.8%** | 83.4% | 79.2% |
| nested router | 87.8% | **83.0%** | 85.1% | 78.2% | 65.8% | 72.8% | **85.4%** | 78.8% |

## Differences in balanced accuracy (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seen-val pick − DINOv2 clean+diff MLP (main runs) | +2.4* [+1.6, +3.3] | -12.6* [-13.6, -11.6] | -0.7 [-1.7, +0.3] | -7.1* [-8.2, -6.0] | -9.0* [-10.1, -7.8] | -14.7* [-15.8, -13.7] | -1.3* [-2.2, -0.3] | -6.9* [-7.4, -6.5] |
| nested clean pick − DINOv2 clean+diff MLP (main runs) | +2.3* [+1.5, +3.1] | -7.3* [-8.2, -6.4] | -1.6* [-2.6, -0.6] | -5.3* [-6.4, -4.2] | -7.6* [-8.7, -6.5] | -6.5* [-7.5, -5.5] | -2.7* [-3.6, -2.0] | -4.3* [-4.7, -3.9] |
| nested one detector − DINOv2 clean+diff MLP (main runs) | -2.0* [-3.0, -1.1] | +2.1* [+1.1, +3.2] | -0.8 [-1.9, +0.3] | +0.9 [-0.3, +2.1] | +2.1* [+0.8, +3.3] | +3.0* [+1.7, +4.2] | -0.3 [-1.3, +0.7] | +0.9* [+0.4, +1.4] |
| nested router − DINOv2 clean+diff MLP (main runs) | +2.5* [+1.7, +3.3] | +8.2* [+7.2, +9.2] | +3.6* [+2.6, +4.6] | +4.1* [+2.9, +5.3] | +4.3* [+2.9, +5.5] | +4.8* [+3.6, +6.0] | +2.9* [+2.1, +3.7] | +4.6* [+4.1, +5.0] |
| nested router − router | +0.5 [-0.3, +1.3] | +0.6 [-0.3, +1.4] | -2.0* [-2.6, -1.4] | -1.2* [-1.8, -0.7] | +0.6 [-0.4, +1.7] | -1.0* [-1.6, -0.3] | +1.9* [+1.1, +2.7] | -0.4* [-0.7, -0.1] |
| nested clean pick − seen-val pick | -0.1 [-0.9, +0.6] | +5.2* [+4.4, +6.1] | -0.9* [-1.8, -0.0] | +1.8* [+0.9, +2.8] | +1.3* [+0.7, +2.0] | +8.2* [+7.4, +9.0] | -1.5* [-2.3, -0.6] | +2.6* [+2.3, +3.0] |

## Single-detector choices per outer holdout

| Held-out generator | seen-val pick | nested clean pick | nested one detector |
|---|---|---|---|
| BigGAN | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | DINOv2 L24 patch LR |
| VQDM | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L20 MLP |
| SD v1.5 | DINOv2 L12-24 LR | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 patch MLP |
| Wukong | DINOv2 L12-24 LR | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 patch MLP |
| ADM | DINOv2 L12-24 MLP | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 patch MLP |
| GLIDE | DINOv2 L12 MLP | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L20 MLP |
| Midjourney | DINOv2 L12-24 LR | DINOv2 L24 + CLIP MLP | DINOv2 L24 patch LR |

## router choices

| Held-out generator | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 |
|---|---|---|---|---|---|---|---|
| BigGAN | DINOv2 L12-24 MLP | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | DINOv2 L16 patch MLP |
| VQDM | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 MLP | DINOv2 L20 MLP | DINOv2 L16 MLP |
| SD v1.5 | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L12-24 MLP |
| Wukong | DINOv2 L12 MLP | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | DINOv2 L16 patch MLP |
| ADM | DINOv2 L12 MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L24 patch MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP |
| GLIDE | DINOv2 L12 MLP | DINOv2 L16 patch MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | DINOv2 L16 patch MLP |
| Midjourney | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | DINOv2 L12-24 LR |

## nested router choices

| Held-out generator | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 |
|---|---|---|---|---|---|---|---|
| BigGAN | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR |
| VQDM | DINOv2 L12-24 MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | DINOv2 L24 MLP | DINOv2 L12-24 LR | DINOv2 L20 MLP |
| SD v1.5 | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | fusion: DINOv2 clean+diff MLP + CLIP MLP |
| Wukong | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | DINOv2 L24 + CLIP MLP | DINOv2 L24 patch MLP | fusion: DINOv2 clean+diff MLP + CLIP MLP |
| ADM | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 MLP | DINOv2 L24 MLP | DINOv2 L24 + CLIP MLP | DINOv2 L24 + CLIP MLP |
| GLIDE | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 MLP | DINOv2 L20 MLP | DINOv2 L24 patch MLP | fusion: DINOv2 clean+diff MLP + CLIP MLP |
| Midjourney | fusion: DINOv2 clean+diff MLP + CLIP MLP | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L12-24 LR |

## Top 10 candidates by inner clean BA (7-holdout mean)

| Candidate | Inner clean BA | Inner mean (6 std.) | Outer clean BA |
|---|---:|---:|---:|
| fusion: DINOv2 clean+diff MLP + CLIP MLP | 88.9% | 68.9% | 90.0% |
| DINOv2 L24 + CLIP MLP | 87.7% | 72.4% | 88.8% |
| DINOv2 L12-24 LR | 87.3% | 69.2% | 88.5% |
| DINOv2 L20 MLP | 87.2% | 75.6% | 88.6% |
| CLIP MLP | 86.7% | 63.7% | 87.7% |
| DINOv2 L12-24 MLP | 86.6% | 68.1% | 88.3% |
| DINOv2 L24 + CLIP LR | 86.6% | 68.9% | 87.6% |
| DINOv2 L20 LR | 86.5% | 74.6% | 87.6% |
| DINOv2 L16 patch MLP | 84.8% | 73.9% | 86.6% |
| CLIP LR | 84.7% | 62.0% | 86.6% |

Nested one-detector picks across the 7 outer folds: DINOv2 L24 patch MLP ×3, DINOv2 L24 patch LR ×2, DINOv2 L20 MLP ×2.
