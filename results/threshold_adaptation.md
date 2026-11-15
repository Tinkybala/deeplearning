# Threshold adaptation: how much of the degradation loss is a threshold shift?

Fixed detectors, five threshold rules. **clean val** = the main runs' rule (clean seen-generator validation); **degraded val** = re-picked on the seen generators' validation images degraded like the test condition (needs the condition; E4's +recal); **test median** and **test GMM** = label-free, from the unlabelled test batch (one held-out generator, one condition): its median score, or the crossing point of a two-component Gaussian mixture on its logit scores; **oracle** = best threshold on the test labels (upper bound). Mean balanced accuracy over the 7 held-out generators; best non-oracle rule per detector and column in bold.

## Balanced accuracy, 50% real test batches

| Detector | Rule | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 MLP | clean val | 85.3% | 74.7% | 81.5% | 74.1% | 61.6% | 68.1% | 82.5% | 74.2% |
| DINOv2 MLP | degraded val | 85.1% | 77.3% | 82.0% | 74.3% | 61.9% | 68.1% | 83.7% | 74.8% |
| DINOv2 MLP | test median | **85.9%** | **77.9%** | **82.9%** | **75.2%** | **63.3%** | **68.7%** | **84.8%** | 75.6% |
| DINOv2 MLP | test GMM | 85.3% | 77.9% | 82.2% | 74.8% | 62.2% | 68.3% | 83.9% | 75.1% |
| DINOv2 MLP | oracle | 86.4% | 79.0% | 83.4% | 76.0% | 63.9% | 69.6% | 85.4% | 76.4% |
| DINOv2 L20 MLP | clean val | 88.6% | 74.7% | 85.8% | 78.5% | 66.4% | 66.3% | 83.4% | 76.7% |
| DINOv2 L20 MLP | degraded val | 88.5% | 82.8% | 85.4% | 77.9% | **66.4%** | 74.4% | 86.7% | 79.2% |
| DINOv2 L20 MLP | test median | **90.1%** | **84.1%** | **86.7%** | 78.8% | 66.4% | **75.3%** | **88.4%** | 80.2% |
| DINOv2 L20 MLP | test GMM | 90.0% | 84.0% | 86.7% | **78.9%** | 66.3% | 75.1% | 88.4% | 80.2% |
| DINOv2 L20 MLP | oracle | 90.6% | 84.7% | 87.4% | 79.3% | 67.3% | 75.7% | 88.9% | 80.8% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | clean val | 90.0% | 69.8% | 77.8% | 66.8% | 54.5% | 66.2% | 81.1% | 70.9% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | degraded val | 89.5% | 83.6% | 84.2% | 75.0% | 62.6% | 70.6% | 87.5% | 77.6% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | test median | **91.1%** | **84.1%** | 84.9% | **75.8%** | 62.7% | **71.2%** | **88.5%** | 78.3% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | test GMM | 90.8% | 84.0% | **85.0%** | 75.6% | **62.9%** | 70.9% | 88.3% | 78.2% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | oracle | 91.5% | 84.8% | 85.4% | 76.4% | 63.5% | 71.9% | 89.2% | 78.9% |
| DINOv2 L24 LR | clean val | 80.5% | 74.8% | 77.9% | 73.1% | 63.8% | 70.1% | 78.9% | 73.4% |
| DINOv2 L24 LR | degraded val | 80.3% | **75.0%** | **78.6%** | 73.5% | 63.9% | 70.4% | 79.3% | 73.6% |
| DINOv2 L24 LR | test median | **80.7%** | 74.9% | 78.6% | **74.0%** | **64.2%** | **70.9%** | **79.9%** | 73.9% |
| DINOv2 L24 LR | test GMM | 79.9% | 74.4% | 77.9% | 73.3% | 64.0% | 70.5% | 79.6% | 73.3% |
| DINOv2 L24 LR | oracle | 81.4% | 75.9% | 79.3% | 74.3% | 65.2% | 71.5% | 80.8% | 74.6% |
| ResNet-18 squash224 | clean val | 72.2% | 63.3% | 66.4% | 51.7% | 52.4% | 59.0% | 71.5% | 60.8% |
| ResNet-18 squash224 | degraded val | 72.0% | **64.1%** | 71.7% | **52.5%** | **52.9%** | 63.7% | 72.2% | 62.8% |
| ResNet-18 squash224 | test median | **73.7%** | 63.7% | 72.6% | 51.8% | 52.3% | **63.8%** | **74.2%** | 63.0% |
| ResNet-18 squash224 | test GMM | 73.6% | 63.6% | **72.8%** | 51.5% | 52.4% | 63.6% | 73.4% | 62.9% |
| ResNet-18 squash224 | oracle | 74.9% | 65.2% | 73.8% | 53.8% | 54.6% | 64.7% | 74.6% | 64.5% |

## Gain over the clean-val threshold (pp), paired bootstrap (2000 replicates per holdout, test-batch thresholds held fixed); * = 95% interval excludes 0

| Detector | Rule − clean val | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 MLP | degraded val | -0.2 [-0.6, +0.1] | +2.5* [+1.8, +3.2] | +0.5* [+0.0, +1.0] | +0.2 [-0.4, +0.8] | +0.3 [-0.4, +1.1] | +0.0 [-0.4, +0.5] | +1.2* [+0.7, +1.8] | +0.6* [+0.3, +0.8] |
| DINOv2 MLP | test median | +0.6* [+0.1, +1.1] | +3.1* [+2.3, +4.0] | +1.4* [+0.8, +2.0] | +1.1* [+0.6, +1.7] | +1.7* [+0.9, +2.5] | +0.6 [-0.1, +1.3] | +2.3* [+1.6, +2.9] | +1.4* [+1.2, +1.7] |
| DINOv2 MLP | test GMM | +0.0 [-0.4, +0.5] | +3.1* [+2.5, +3.7] | +0.7* [+0.0, +1.3] | +0.7 [-0.0, +1.5] | +0.7 [-0.3, +1.6] | +0.3 [-0.3, +0.8] | +1.4* [+0.9, +1.9] | +0.9* [+0.6, +1.2] |
| DINOv2 MLP | oracle | +1.1* [+0.7, +1.6] | +4.3* [+3.5, +5.0] | +1.9* [+1.3, +2.5] | +1.9* [+1.3, +2.4] | +2.4* [+1.6, +3.2] | +1.5* [+0.8, +2.3] | +2.9* [+2.3, +3.7] | +2.2* [+1.9, +2.4] |
| DINOv2 L20 MLP | degraded val | -0.2 [-0.4, +0.1] | +8.1* [+7.3, +8.9] | -0.5 [-0.9, +0.0] | -0.6 [-1.3, +0.1] | +0.0 [-0.6, +0.7] | +8.0* [+7.1, +9.0] | +3.3* [+2.7, +3.8] | +2.5* [+2.2, +2.8] |
| DINOv2 L20 MLP | test median | +1.5* [+0.9, +2.0] | +9.4* [+8.5, +10.3] | +0.9* [+0.4, +1.4] | +0.3 [-0.2, +0.8] | -0.0 [-0.7, +0.6] | +8.9* [+7.9, +9.9] | +5.0* [+4.3, +5.8] | +3.5* [+3.2, +3.8] |
| DINOv2 L20 MLP | test GMM | +1.3* [+0.8, +1.9] | +9.3* [+8.4, +10.2] | +0.9* [+0.4, +1.3] | +0.5 [-0.1, +1.0] | -0.1 [-0.6, +0.4] | +8.8* [+7.8, +9.8] | +5.0* [+4.3, +5.6] | +3.4* [+3.2, +3.7] |
| DINOv2 L20 MLP | oracle | +2.0* [+1.4, +2.5] | +10.0* [+9.0, +10.9] | +1.6* [+1.0, +2.2] | +0.9* [+0.3, +1.4] | +0.9* [+0.2, +1.5] | +9.4* [+8.4, +10.3] | +5.5* [+4.8, +6.2] | +4.1* [+3.8, +4.4] |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | degraded val | -0.5* [-0.8, -0.2] | +13.7* [+12.8, +14.7] | +6.4* [+5.4, +7.3] | +8.2* [+7.2, +9.3] | +8.2* [+7.0, +9.3] | +4.4* [+3.5, +5.3] | +6.5* [+5.7, +7.2] | +6.7* [+6.4, +7.1] |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | test median | +1.1* [+0.6, +1.6] | +14.3* [+13.3, +15.3] | +7.1* [+6.3, +8.0] | +9.0* [+8.0, +10.0] | +8.2* [+7.1, +9.4] | +5.0* [+3.9, +6.0] | +7.5* [+6.6, +8.3] | +7.4* [+7.1, +7.8] |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | test GMM | +0.8* [+0.4, +1.2] | +14.1* [+13.2, +15.1] | +7.2* [+6.2, +8.0] | +8.8* [+7.8, +9.9] | +8.4* [+7.3, +9.6] | +4.7* [+3.8, +5.6] | +7.2* [+6.5, +8.0] | +7.3* [+7.0, +7.7] |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | oracle | +1.5* [+1.0, +2.0] | +15.0* [+14.0, +15.9] | +7.6* [+6.8, +8.5] | +9.5* [+8.6, +10.5] | +9.1* [+7.9, +10.2] | +5.7* [+4.7, +6.7] | +8.1* [+7.3, +9.0] | +8.1* [+7.7, +8.4] |
| DINOv2 L24 LR | degraded val | -0.2 [-0.6, +0.3] | +0.2 [-0.2, +0.7] | +0.7* [+0.2, +1.2] | +0.4 [-0.3, +1.1] | +0.1 [-0.6, +0.8] | +0.2 [-0.4, +0.9] | +0.4 [-0.2, +0.9] | +0.2* [+0.0, +0.5] |
| DINOv2 L24 LR | test median | +0.2 [-0.4, +0.8] | +0.1 [-0.5, +0.6] | +0.7* [+0.1, +1.2] | +0.9* [+0.2, +1.5] | +0.4 [-0.3, +1.1] | +0.8* [+0.1, +1.4] | +1.0* [+0.4, +1.6] | +0.5* [+0.2, +0.8] |
| DINOv2 L24 LR | test GMM | -0.6* [-1.3, -0.0] | -0.4 [-1.1, +0.2] | -0.0 [-0.7, +0.6] | +0.2 [-0.5, +1.0] | +0.1 [-0.7, +1.0] | +0.4 [-0.1, +0.9] | +0.7* [+0.1, +1.2] | -0.1 [-0.3, +0.2] |
| DINOv2 L24 LR | oracle | +0.9* [+0.4, +1.4] | +1.1* [+0.6, +1.5] | +1.4* [+0.8, +2.0] | +1.3* [+0.7, +1.8] | +1.4* [+0.8, +2.1] | +1.4* [+0.9, +2.0] | +1.9* [+1.2, +2.5] | +1.2* [+1.0, +1.5] |
| ResNet-18 squash224 | degraded val | -0.2 [-0.5, +0.2] | +0.8* [+0.1, +1.5] | +5.3* [+4.3, +6.4] | +0.7 [-0.3, +1.9] | +0.5 [-0.5, +1.5] | +4.6* [+3.6, +5.7] | +0.7* [+0.4, +1.0] | +2.0* [+1.6, +2.3] |
| ResNet-18 squash224 | test median | +1.5* [+0.7, +2.3] | +0.4 [-0.5, +1.4] | +6.2* [+5.2, +7.2] | +0.1 [-1.0, +1.3] | -0.1 [-1.1, +0.9] | +4.8* [+3.8, +5.8] | +2.7* [+1.9, +3.5] | +2.2* [+1.7, +2.6] |
| ResNet-18 squash224 | test GMM | +1.4* [+0.7, +2.1] | +0.3 [-0.5, +1.1] | +6.4* [+5.5, +7.3] | -0.2 [-1.3, +1.0] | -0.0 [-1.1, +1.0] | +4.6* [+3.6, +5.5] | +1.9* [+1.2, +2.5] | +2.1* [+1.7, +2.5] |
| ResNet-18 squash224 | oracle | +2.7* [+2.0, +3.4] | +1.9* [+1.1, +2.7] | +7.4* [+6.5, +8.3] | +2.1* [+1.2, +3.0] | +2.2* [+1.2, +3.2] | +5.7* [+4.7, +6.7] | +3.1* [+2.3, +3.9] | +3.7* [+3.3, +4.0] |

## Sensitivity to the share of real images in the test batch: mean BA over the six standard conditions (20 random minority subsets per skewed batch)

| Detector | Rule | 50% real | 20% real | 80% real |
|---|---|---:|---:|---:|
| DINOv2 MLP | clean val | 74.2% | 74.1% | 74.3% |
| DINOv2 MLP | degraded val | 74.8% | 74.7% | 74.7% |
| DINOv2 MLP | test median | 75.6% | 72.4% | 71.7% |
| DINOv2 MLP | test GMM | 75.1% | 72.4% | 75.5% |
| DINOv2 MLP | oracle | 76.4% | 76.9% | 76.8% |
| DINOv2 L20 MLP | clean val | 76.7% | 76.7% | 76.8% |
| DINOv2 L20 MLP | degraded val | 79.2% | 79.2% | 79.3% |
| DINOv2 L20 MLP | test median | 80.2% | 74.9% | 75.2% |
| DINOv2 L20 MLP | test GMM | 80.2% | 78.7% | 79.7% |
| DINOv2 L20 MLP | oracle | 80.8% | 81.3% | 81.4% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | clean val | 70.9% | 70.8% | 70.9% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | degraded val | 77.6% | 77.5% | 77.6% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | test median | 78.3% | 73.8% | 73.6% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | test GMM | 78.2% | 76.4% | 77.9% |
| fusion: DINOv2 clean+diff MLP + CLIP MLP | oracle | 78.9% | 79.5% | 79.5% |
| DINOv2 L24 LR | clean val | 73.4% | 73.4% | 73.4% |
| DINOv2 L24 LR | degraded val | 73.6% | 73.6% | 73.6% |
| DINOv2 L24 LR | test median | 73.9% | 71.2% | 71.4% |
| DINOv2 L24 LR | test GMM | 73.3% | 71.1% | 73.8% |
| DINOv2 L24 LR | oracle | 74.6% | 75.2% | 75.2% |
| ResNet-18 squash224 | clean val | 60.8% | 60.9% | 60.9% |
| ResNet-18 squash224 | degraded val | 62.8% | 62.8% | 62.8% |
| ResNet-18 squash224 | test median | 63.0% | 62.0% | 61.8% |
| ResNet-18 squash224 | test GMM | 62.9% | 61.9% | 62.8% |
| ResNet-18 squash224 | oracle | 64.5% | 65.3% | 65.3% |

## Class recall (real / generated), 50% real batches

| Detector | Rule | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 MLP | clean val | 87% / 83% | 92% / 57% | 82% / 81% | 71% / 77% | 48% / 76% | 79% / 58% | 93% / 72% |
| DINOv2 MLP | degraded val | 90% / 81% | 83% / 71% | 86% / 78% | 79% / 70% | 62% / 61% | 76% / 60% | 88% / 80% |
| DINOv2 MLP | test median | 86% / 86% | 78% / 78% | 83% / 83% | 75% / 75% | 63% / 63% | 69% / 69% | 85% / 85% |
| DINOv2 MLP | test GMM | 90% / 80% | 88% / 68% | 90% / 75% | 83% / 67% | 71% / 53% | 78% / 58% | 91% / 77% |
| DINOv2 MLP | oracle | 88% / 85% | 83% / 75% | 85% / 82% | 80% / 72% | 63% / 65% | 71% / 69% | 86% / 85% |
| ResNet-18 squash224 | clean val | 86% / 58% | 88% / 39% | 42% / 91% | 11% / 93% | 20% / 85% | 88% / 30% | 88% / 55% |
| ResNet-18 squash224 | degraded val | 86% / 58% | 79% / 49% | 78% / 65% | 58% / 47% | 55% / 51% | 69% / 59% | 87% / 58% |
| ResNet-18 squash224 | test median | 74% / 74% | 64% / 64% | 73% / 73% | 52% / 52% | 52% / 52% | 64% / 64% | 74% / 74% |
| ResNet-18 squash224 | test GMM | 80% / 67% | 75% / 53% | 70% / 76% | 52% / 52% | 50% / 55% | 77% / 50% | 80% / 67% |
| ResNet-18 squash224 | oracle | 80% / 70% | 71% / 60% | 65% / 82% | 48% / 60% | 60% / 49% | 67% / 63% | 76% / 73% |
