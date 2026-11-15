# Final detector: block-20 probes, augmented training, batch and single-image thresholds

Mean balanced accuracy over the 7 held-out generators. Training: **[both]** = fit on the augmented images only (JPEG 30–95 and/or resize to 32/64/128 px, p = 0.5 each), **[clean+both]** = clean and augmented together, no tag = clean only. Thresholds: **clean val** (main runs), **degraded val** (needs the test condition), **metadata A** (single image: threshold of its bucket = shorter side < 48 / < 96 / full and JPEG quality <= 75, learned on degraded seen-generator validation), **metadata B** (size buckets only), **test GMM** (label-free, needs a batch of test images of one condition), **oracle** (test labels; upper bound). Routers choose among every probe here and in backbone_comparison, per condition (needs the condition) or per metadata bucket.

## DINOv2 clean+diff MLP (main runs)

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean val | 85.3% | 74.7% | 81.5% | 74.1% | 61.6% | 68.1% | 82.5% | 74.2% |
| degraded val | 85.1% | 77.3% | 82.0% | 74.3% | 61.9% | 68.1% | 83.7% | 74.8% |
| metadata A | 85.1% | 77.2% | 81.7% | 74.4% | 61.9% | 67.6% | 82.6% | 74.6% |
| metadata B | 85.5% | 76.2% | 81.4% | 74.4% | 61.9% | 68.3% | 83.2% | 74.6% |
| test GMM | 85.3% | 77.9% | 82.2% | 74.8% | 62.2% | 68.3% | 83.9% | 75.1% |
| oracle | 86.4% | 79.0% | 83.4% | 76.0% | 63.9% | 69.6% | 85.4% | 76.4% |

## DINOv2 clean+diff MLP (main runs) [clean+both]

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean val | 84.2% | 78.3% | 81.5% | 75.8% | 65.0% | 70.5% | 81.9% | 75.9% |
| degraded val | 84.3% | 79.3% | 82.2% | 76.2% | 66.3% | 71.1% | 84.0% | 76.6% |
| metadata A | 84.5% | 78.7% | 81.9% | 76.2% | 66.3% | 70.9% | 82.5% | 76.4% |
| metadata B | 84.5% | 78.9% | 82.0% | 76.2% | 66.3% | 71.0% | 82.2% | 76.5% |
| test GMM | 84.6% | 79.0% | 81.9% | 76.2% | 66.1% | 71.2% | 83.9% | 76.5% |
| oracle | 85.8% | 80.4% | 83.1% | 77.5% | 67.3% | 72.3% | 85.1% | 77.7% |

## DINOv2 L20 MLP

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean val | 88.6% | 74.7% | 85.8% | 78.5% | 66.4% | 66.3% | 83.4% | 76.7% |
| degraded val | 88.5% | 82.8% | 85.4% | 77.9% | 66.4% | 74.4% | 86.7% | 79.2% |
| metadata A | 88.9% | 83.0% | 85.9% | 78.0% | 66.4% | 74.3% | 84.5% | 79.4% |
| metadata B | 88.6% | 80.0% | 84.8% | 78.1% | 66.4% | 70.9% | 86.8% | 78.2% |
| test GMM | 90.0% | 84.0% | 86.7% | 78.9% | 66.3% | 75.1% | 88.4% | 80.2% |
| oracle | 90.6% | 84.7% | 87.4% | 79.3% | 67.3% | 75.7% | 88.9% | 80.8% |

## DINOv2 L20 MLP [both]

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean val | 88.5% | 83.3% | 85.9% | 79.4% | 71.5% | 76.6% | 86.0% | 80.9% |
| degraded val | 88.0% | 83.6% | 84.9% | 79.0% | 71.5% | 76.3% | 87.5% | 80.5% |
| metadata A | 88.4% | 83.2% | 85.7% | 79.0% | 71.5% | 76.7% | 86.3% | 80.7% |
| metadata B | 88.3% | 83.5% | 85.6% | 79.0% | 71.5% | 76.7% | 86.2% | 80.8% |
| test GMM | 89.0% | 85.1% | 86.5% | 80.0% | 72.2% | 77.7% | 88.4% | 81.8% |
| oracle | 89.6% | 85.6% | 86.9% | 80.7% | 72.9% | 78.1% | 89.1% | 82.3% |

## DINOv2 L20 MLP [clean+both]

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean val | 88.7% | 83.2% | 86.7% | 80.7% | 72.3% | 76.4% | 85.5% | 81.3% |
| degraded val | 88.6% | 84.8% | 86.6% | 80.5% | 72.4% | 76.8% | 87.4% | 81.6% |
| metadata A | 89.2% | 83.9% | 87.1% | 80.5% | 72.4% | 76.7% | 86.1% | 81.6% |
| metadata B | 89.4% | 83.8% | 87.1% | 80.5% | 72.4% | 77.1% | 86.3% | 81.7% |
| test GMM | 90.5% | 86.2% | 88.1% | 81.6% | 72.8% | 78.0% | 89.0% | 82.9% |
| oracle | 91.0% | 86.7% | 88.6% | 82.0% | 73.5% | 78.7% | 89.6% | 83.4% |

## fusion: DINOv2 clean+diff MLP + CLIP MLP

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean val | 90.0% | 69.8% | 77.8% | 66.8% | 54.5% | 66.2% | 81.1% | 70.9% |
| degraded val | 89.5% | 83.6% | 84.2% | 75.0% | 62.6% | 70.6% | 87.5% | 77.6% |
| metadata A | 88.9% | 82.6% | 80.9% | 75.1% | 62.6% | 70.5% | 78.0% | 76.8% |
| metadata B | 89.4% | 76.8% | 70.6% | 75.1% | 62.6% | 69.6% | 85.4% | 74.0% |
| test GMM | 90.8% | 84.0% | 85.0% | 75.6% | 62.9% | 70.9% | 88.3% | 78.2% |
| oracle | 91.5% | 84.8% | 85.4% | 76.4% | 63.5% | 71.9% | 89.2% | 78.9% |

## Routers

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| condition router | 87.3% | 84.8% | 87.1% | 79.3% | 71.9% | 76.8% | 83.4% | 81.2% |
| metadata router A | 86.7% | 83.9% | 85.6% | 79.4% | 71.9% | 76.7% | 83.7% | 80.7% |
| metadata router B | 89.4% | 83.8% | 87.1% | 79.4% | 71.9% | 77.1% | 86.3% | 81.5% |

## Differences in balanced accuracy (pp), paired bootstrap (2000 replicates per holdout; batch thresholds held fixed); * = 95% interval excludes 0

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 L20 MLP [clean+both], clean val − DINOv2 L20 MLP, clean val | +0.1 [-0.4, +0.6] | +8.5* [+7.7, +9.2] | +0.9* [+0.3, +1.5] | +2.2* [+1.4, +3.0] | +5.9* [+4.8, +6.9] | +10.0* [+9.0, +11.0] | +2.2* [+1.6, +2.7] | +4.6* [+4.3, +4.9] |
| DINOv2 L20 MLP [clean+both], test GMM − DINOv2 L20 MLP, test GMM | +0.6* [+0.1, +1.1] | +2.2* [+1.5, +2.9] | +1.4* [+0.8, +2.0] | +2.6* [+1.8, +3.4] | +6.5* [+5.4, +7.6] | +2.8* [+2.1, +3.7] | +0.6* [+0.1, +1.1] | +2.7* [+2.4, +3.0] |
| DINOv2 L20 MLP [clean+both], metadata A − DINOv2 L20 MLP, test GMM | -0.7* [-1.3, -0.2] | -0.1 [-0.8, +0.7] | +0.4 [-0.2, +1.0] | +1.6* [+0.8, +2.4] | +6.1* [+5.0, +7.2] | +1.6* [+0.7, +2.5] | -2.2* [-2.9, -1.6] | +1.5* [+1.1, +1.8] |
| DINOv2 L20 MLP [clean+both], metadata B − DINOv2 L20 MLP, test GMM | -0.6* [-1.1, -0.0] | -0.2 [-0.9, +0.6] | +0.4 [-0.3, +1.0] | +1.6* [+0.8, +2.4] | +6.1* [+5.0, +7.2] | +1.9* [+1.1, +2.8] | -2.1* [-2.8, -1.5] | +1.5* [+1.2, +1.9] |
| DINOv2 L20 MLP [clean+both], metadata A − condition router | +1.9* [+1.2, +2.7] | -0.9* [-1.2, -0.6] | +0.0 [-0.8, +0.8] | +1.2* [+0.7, +1.7] | +0.5* [+0.1, +0.9] | -0.0 [-0.5, +0.4] | +2.7* [+1.9, +3.5] | +0.4* [+0.2, +0.7] |
| metadata router A − condition router | -0.6 [-1.4, +0.1] | -0.9* [-1.2, -0.5] | -1.5* [-2.3, -0.8] | +0.1 [-0.0, +0.2] | +0.0 [+0.0, +0.0] | -0.0 [-0.5, +0.4] | +0.3 [-0.4, +0.9] | -0.5* [-0.7, -0.3] |
| metadata router B − condition router | +2.1* [+1.3, +2.8] | -0.9* [-1.3, -0.6] | -0.0 [-0.8, +0.7] | +0.1 [-0.0, +0.2] | +0.0 [+0.0, +0.0] | +0.3 [-0.1, +0.8] | +2.8* [+2.0, +3.6] | +0.3* [+0.1, +0.5] |
| DINOv2 L20 MLP [clean+both], test GMM − condition router | +3.2* [+2.4, +4.0] | +1.4* [+0.8, +2.1] | +1.0* [+0.2, +1.8] | +2.2* [+1.5, +3.0] | +0.9* [+0.2, +1.5] | +1.2* [+0.6, +1.8] | +5.5* [+4.7, +6.4] | +1.7* [+1.4, +1.9] |

## Test GMM on skewed batches: mean BA over the six standard conditions

| Detector | 50% real | 20% real | 80% real |
|---|---:|---:|---:|
| DINOv2 L20 MLP | 80.2% | 78.7% | 79.6% |
| DINOv2 L20 MLP [both] | 81.8% | 80.5% | 80.7% |
| DINOv2 L20 MLP [clean+both] | 82.9% | 81.6% | 81.7% |

## Metadata A buckets: test images per bucket, real / generated (all 7 holdouts)

| Condition | <48 | <96 | <96, jpeg<=75 | full | full, jpeg<=75 |
|---|---:|---:|---:|---:|---:|
| Clean |  | 5 / 0 | 5 / 0 | 3357 / 3500 | 133 / 0 |
| JPEG Q50 |  |  | 10 / 0 |  | 3490 / 3500 |
| Resize 128 |  |  |  | 3500 / 3500 |  |
| Resize 64 |  | 3500 / 3500 |  |  |  |
| Resize 32 | 3500 / 3500 |  |  |  |  |
| 128 + JPEG 50 |  |  |  |  | 3500 / 3500 |
| Format 256+q96 |  |  |  | 3500 / 3500 |  |

## Metadata router A choices (bucket: probe)

| Held-out generator | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 |
|---|---|---|---|---|---|---|---|
| BigGAN | <96, jpeg<=75: DINOv2 L20 MLP; full: DINOv2 L12-24 LR; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | <96, jpeg<=75: DINOv2 L20 MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L12-24 LR | <96: DINOv2 L12-24 MLP | <48: DINOv2 L20 MLP [clean+both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L12-24 LR |
| VQDM | <96: DINOv2 L20 MLP [clean+both]; <96, jpeg<=75: DINOv2 L20 MLP; full: DINOv2 L16 patch MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | <96, jpeg<=75: DINOv2 L20 MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP | <96: DINOv2 L20 MLP [clean+both] | <48: DINOv2 L20 MLP [both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP |
| SD v1.5 | <96: DINOv2 L20 MLP [clean+both]; <96, jpeg<=75: DINOv2 L20 MLP; full: DINOv2 L16 patch MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | <96, jpeg<=75: DINOv2 L20 MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP | <96: DINOv2 L20 MLP [clean+both] | <48: DINOv2 L20 MLP [clean+both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP |
| Wukong | <96: DINOv2 L20 MLP [clean+both]; full: DINOv2 L16 patch MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | <96, jpeg<=75: DINOv2 L20 MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP | <96: DINOv2 L20 MLP [clean+both] | <48: DINOv2 L20 MLP [both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP |
| ADM | full: DINOv2 L16 patch MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP | <96: DINOv2 L20 MLP [clean+both] | <48: DINOv2 L20 MLP [clean+both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP |
| GLIDE | <96, jpeg<=75: DINOv2 L20 MLP; full: DINOv2 L16 patch MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | <96, jpeg<=75: DINOv2 L20 MLP; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP | <96: DINOv2 L20 MLP [clean+both] | <48: DINOv2 L20 MLP [clean+both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L16 patch MLP |
| Midjourney | full: DINOv2 L12-24 LR; full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L12-24 LR | <96: DINOv2 L12-24 LR | <48: DINOv2 L20 MLP [clean+both] | full, jpeg<=75: DINOv2 L20 MLP [clean+both] | full: DINOv2 L12-24 LR |
