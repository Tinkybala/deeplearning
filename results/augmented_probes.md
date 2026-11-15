# Degradation-augmented DINOv2 probes (all 7 holdouts)

Probes fit on DINOv2 embeddings of the seen generators' training images after random artifacts applied once per image: **jpeg** = JPEG quality 30–95 (p = 0.5); **resize** = downsample to 32/64/128 px and back to 256 (p = 0.5); **both** = both, independently; **clean+both** = clean and *both* embeddings together. **[clean]** = the main runs' probes. Thresholds from the clean seen-generator validation split, so every row except the routers and *+recal* is a single detector that is not told the test condition. **+recal** = threshold re-picked on the seen generators' validation images degraded like the test condition; **router** = per condition, the probe and threshold best on those images (E4), from the E4 pool (clean MLP, clean logistic regression, their fusion) or from every probe here. Mean over the 7 held-out generators; best per column in bold.

Reproduction: on the GLIDE holdout the jpeg / resize / both MLPs match the earlier GLIDE-only augmented runs to within 0.00 pp on every standard condition.

## Single detectors (clean seen-val threshold), balanced accuracy

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff [clean] | **85.3%** | 74.7% | **81.5%** | 74.1% | 61.6% | 68.1% | **83.3%** | **82.5%** | 74.2% |
| mlp_clean+diff [jpeg] | 84.0% | **79.3%** | 80.8% | 73.7% | 61.8% | 69.2% | 82.6% | 81.9% | 74.8% |
| mlp_clean+diff [resize] | 83.1% | 75.7% | 80.5% | 75.5% | 65.6% | 69.5% | 81.0% | 80.6% | 75.0% |
| mlp_clean+diff [both] | 82.7% | 78.3% | 80.8% | **75.8%** | 65.6% | **71.5%** | 81.0% | 80.7% | 75.8% |
| mlp_clean+diff [clean+both] | 84.2% | 78.3% | **81.5%** | 75.8% | 65.0% | 70.5% | 82.5% | 81.9% | 75.9% |
| logreg_clean [clean] | 80.5% | 74.8% | 77.9% | 73.1% | 63.8% | 70.1% | 78.5% | 78.9% | 73.4% |
| logreg_clean [jpeg] | 80.6% | 76.0% | 78.3% | 73.7% | 64.8% | 71.5% | 79.2% | 79.4% | 74.1% |
| logreg_clean [resize] | 79.9% | 74.6% | 77.6% | 73.8% | 66.3% | 70.0% | 78.1% | 78.4% | 73.7% |
| logreg_clean [both] | 79.2% | 75.1% | 77.6% | 73.9% | **66.7%** | 71.4% | 78.0% | 78.6% | 74.0% |
| logreg_clean [clean+both] | 80.9% | 76.5% | 79.0% | 74.4% | 66.6% | 71.2% | 79.3% | 79.7% | 74.8% |
| fusion [clean] | 83.5% | 76.5% | 80.3% | 74.0% | 63.8% | 70.7% | 81.5% | 81.2% | 74.8% |

## AUROC

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff [clean] | **0.927** | 0.854 | **0.901** | 0.823 | 0.669 | 0.750 | **0.911** | **0.916** | 0.821 |
| mlp_clean+diff [jpeg] | 0.920 | **0.874** | 0.898 | 0.829 | 0.687 | 0.771 | 0.907 | 0.911 | 0.830 |
| mlp_clean+diff [resize] | 0.914 | 0.854 | 0.891 | 0.834 | 0.720 | 0.773 | 0.893 | 0.900 | 0.831 |
| mlp_clean+diff [both] | 0.906 | 0.857 | 0.884 | 0.834 | **0.725** | **0.785** | 0.892 | 0.895 | 0.832 |
| mlp_clean+diff [clean+both] | 0.922 | 0.874 | 0.899 | **0.840** | 0.719 | 0.783 | 0.908 | 0.913 | 0.840 |

## Condition-aware variants, balanced accuracy

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff [clean] +recal | **85.1%** | 77.3% | 82.0% | 74.3% | 61.9% | 68.1% | **83.6%** | 83.7% | 74.8% |
| mlp_clean+diff [jpeg] +recal | 84.2% | 79.3% | 81.1% | 74.7% | 63.2% | 69.8% | 82.9% | 83.0% | 75.4% |
| mlp_clean+diff [resize] +recal | 83.0% | 77.4% | 80.7% | 75.5% | 65.8% | 70.3% | 81.3% | 82.0% | 75.4% |
| mlp_clean+diff [both] +recal | 82.4% | 78.2% | 81.1% | 75.2% | **66.9%** | 71.1% | 81.1% | 82.0% | 75.8% |
| mlp_clean+diff [clean+both] +recal | 84.3% | **79.3%** | **82.2%** | **76.2%** | 66.3% | 71.1% | 82.7% | **84.0%** | 76.6% |
| router (E4 pool) | **85.1%** | 76.3% | 82.0% | 74.8% | 64.6% | 71.1% | **83.6%** | 83.7% | 75.7% |
| router (+ augmented probes) | **85.1%** | 79.3% | **82.2%** | **76.2%** | 66.4% | **71.7%** | 82.9% | 83.6% | 76.8% |

## Differences in balanced accuracy (pp), 7-generator mean, paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff [jpeg] − mlp_clean+diff [clean] | -1.3* [-1.8, -0.7] | +4.5* [+3.7, +5.3] | -0.7* [-1.3, -0.1] | -0.4 [-1.1, +0.4] | +0.3 [-0.6, +1.1] | +1.1* [+0.2, +2.0] | -0.8* [-1.4, -0.2] | -0.6* [-1.1, -0.1] | +0.6* [+0.3, +0.9] |
| mlp_clean+diff [resize] − mlp_clean+diff [clean] | -2.2* [-2.8, -1.6] | +1.0* [+0.3, +1.7] | -1.1* [-1.7, -0.3] | +1.4* [+0.6, +2.2] | +4.0* [+3.1, +4.9] | +1.4* [+0.6, +2.3] | -2.3* [-2.9, -1.7] | -1.9* [-2.4, -1.3] | +0.8* [+0.4, +1.1] |
| mlp_clean+diff [both] − mlp_clean+diff [clean] | -2.6* [-3.4, -1.9] | +3.5* [+2.7, +4.3] | -0.7 [-1.5, +0.1] | +1.7* [+0.8, +2.6] | +4.0* [+3.0, +5.0] | +3.4* [+2.5, +4.4] | -2.3* [-3.0, -1.5] | -1.8* [-2.6, -1.2] | +1.6* [+1.2, +1.9] |
| mlp_clean+diff [clean+both] − mlp_clean+diff [clean] | -1.1* [-1.8, -0.5] | +3.6* [+2.9, +4.4] | +0.0 [-0.7, +0.7] | +1.7* [+0.9, +2.5] | +3.4* [+2.4, +4.4] | +2.5* [+1.6, +3.4] | -0.8* [-1.4, -0.1] | -0.6* [-1.2, -0.0] | +1.7* [+1.4, +2.0] |
| mlp_clean+diff [both] − router (E4 pool) | -2.4* [-3.1, -1.7] | +2.0* [+1.1, +2.9] | -1.2* [-2.0, -0.4] | +1.1* [+0.1, +2.0] | +0.9 [-0.2, +2.0] | +0.3 [-0.7, +1.3] | -2.5* [-3.3, -1.8] | -3.1* [-3.9, -2.3] | +0.1 [-0.3, +0.5] |
| mlp_clean+diff [clean+both] − router (E4 pool) | -0.9* [-1.5, -0.3] | +2.0* [+1.3, +2.8] | -0.5 [-1.2, +0.2] | +1.1* [+0.2, +2.0] | +0.4 [-0.7, +1.4] | -0.6 [-1.5, +0.3] | -1.0* [-1.7, -0.3] | -1.9* [-2.6, -1.1] | +0.2 [-0.1, +0.6] |
| router (+ augmented probes) − router (E4 pool) | +0.0 [+0.0, +0.0] | +3.1* [+2.2, +3.8] | +0.2 [-0.5, +0.8] | +1.5* [+0.6, +2.4] | +1.7* [+0.7, +2.8] | +0.6 [-0.3, +1.5] | -0.6* [-1.2, -0.1] | -0.1 [-0.5, +0.3] | +1.2* [+0.8, +1.5] |

## Class recall (7-generator mean): real / generated

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff [clean] | 87% / 83% | 92% / 57% | 82% / 81% | 71% / 77% | 48% / 76% | 79% / 58% | 88% / 79% | 93% / 72% |
| mlp_clean+diff [jpeg] | 87% / 82% | 85% / 74% | 80% / 82% | 69% / 79% | 42% / 82% | 63% / 75% | 87% / 79% | 92% / 72% |
| mlp_clean+diff [resize] | 87% / 79% | 89% / 62% | 83% / 78% | 76% / 75% | 52% / 79% | 82% / 57% | 87% / 75% | 92% / 69% |
| mlp_clean+diff [both] | 88% / 77% | 86% / 71% | 84% / 77% | 77% / 75% | 52% / 79% | 74% / 69% | 88% / 74% | 92% / 69% |
| mlp_clean+diff [clean+both] | 89% / 80% | 87% / 70% | 83% / 80% | 75% / 77% | 52% / 78% | 70% / 71% | 89% / 76% | 94% / 70% |

## Clean balanced accuracy per held-out generator

| Held-out generator | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [resize] | mlp_clean+diff [both] | mlp_clean+diff [clean+both] | router (E4 pool) |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | **92.8%** | 91.2% | 90.7% | 91.6% | 91.2% | 92.6% |
| VQDM | **86.1%** | 84.2% | 83.1% | 85.1% | 85.6% | **86.1%** |
| SD v1.5 | **85.6%** | 84.2% | 83.0% | 80.7% | 84.1% | 84.8% |
| Wukong | **86.5%** | 85.1% | 84.8% | 82.7% | 86.1% | 85.9% |
| ADM | **76.6%** | 75.9% | 74.8% | 75.0% | 75.3% | 76.2% |
| GLIDE | 91.5% | 90.4% | 89.9% | 89.9% | 91.0% | **92.5%** |
| Midjourney | **78.1%** | 77.3% | 75.7% | 73.8% | 76.2% | 77.6% |

## JPEG Q50 balanced accuracy per held-out generator

| Held-out generator | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [resize] | mlp_clean+diff [both] | mlp_clean+diff [clean+both] | router (E4 pool) |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | 84.6% | 86.9% | 85.1% | **87.1%** | 86.5% | 83.8% |
| VQDM | 79.8% | 82.8% | 80.4% | 83.9% | **84.7%** | 78.9% |
| SD v1.5 | 76.8% | **78.9%** | 76.4% | 76.4% | 77.7% | 76.7% |
| Wukong | 76.9% | 80.0% | 78.3% | 79.9% | 78.6% | **80.1%** |
| ADM | 64.2% | **73.0%** | 65.0% | 72.8% | 70.6% | 68.2% |
| GLIDE | 76.7% | **85.2%** | 79.1% | 82.1% | 84.4% | 80.1% |
| Midjourney | 64.2% | **68.0%** | 65.9% | 65.7% | 65.7% | 66.1% |

## Resize 64 balanced accuracy per held-out generator

| Held-out generator | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [resize] | mlp_clean+diff [both] | mlp_clean+diff [clean+both] | router (E4 pool) |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | 81.6% | 77.6% | 83.0% | **84.5%** | 81.0% | 81.0% |
| VQDM | 76.2% | 76.8% | 77.5% | 78.6% | 78.5% | **80.2%** |
| SD v1.5 | 70.8% | 71.1% | 72.5% | 73.8% | **74.4%** | 70.9% |
| Wukong | 74.2% | 75.6% | 77.2% | **78.4%** | 78.3% | 77.8% |
| ADM | **72.2%** | **72.2%** | 72.0% | 71.2% | 71.9% | 69.3% |
| GLIDE | 75.5% | 75.7% | 76.8% | 76.4% | 78.1% | **79.2%** |
| Midjourney | 68.2% | 67.0% | **69.5%** | 68.0% | 68.6% | 64.9% |

## Router (+ augmented probes) choice

| Held-out generator | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---|---|---|---|---|---|---|---|
| BigGAN | mlp_clean+diff [clean] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | logreg_clean [both] | logreg_clean [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] |
| VQDM | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | logreg_clean [both] | logreg_clean [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean] |
| SD v1.5 | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [both] | logreg_clean [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean] |
| Wukong | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [both] | logreg_clean [both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean] |
| ADM | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | logreg_clean [both] | logreg_clean [clean+both] | mlp_clean+diff [clean] | mlp_clean+diff [clean] |
| GLIDE | mlp_clean+diff [clean] | mlp_clean+diff [jpeg] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | logreg_clean [both] | logreg_clean [both] | mlp_clean+diff [clean] | mlp_clean+diff [clean+both] |
| Midjourney | mlp_clean+diff [clean] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] | logreg_clean [both] | mlp_clean+diff [clean+both] | mlp_clean+diff [clean+both] |
