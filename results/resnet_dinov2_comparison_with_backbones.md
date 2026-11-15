# Fine-tuned ResNet-18 vs frozen DINOv2, and their combination

ResNet-18 = Tyrone's seven leave-one-generator-out checkpoints (ImageNet-pretrained, fine-tuned end to end on the six seen generators with Resize 256 + RandomResizedCrop 224, no degradation augmentation). **squash224** = Resize((224, 224)) at test time, as in his published results; **crop** = Resize 256 + CenterCrop 224, the view the DINOv2 probes see. **DINOv2 MLP** = the main runs' clean + diff MLP. **Ensemble** = mean of the two models' logits after standardising each on the clean seen-generator validation split. **Router** = per condition, the method and threshold best on the six seen generators' validation images degraded the same way (pool: the rows above plus the 22 frozen-feature probes of backbone_comparison). Thresholds of every other row come from the clean seen-generator validation split. Same split, test images and conditions throughout; mean over the 7 held-out generators, best per column in bold.

Reproduction: with Tyrone's own checkpoint thresholds, the squash224 scores computed here give his published balanced accuracy on every holdout and standard condition to within 0.10 pp (GPU vs Apple-silicon arithmetic).

## Balanced accuracy

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 | Mean (6 std.) | Seen val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 squash224 | 72.2% | 63.3% | 66.4% | 51.7% | 52.4% | 59.0% | 70.3% | 71.5% | 60.8% | 86.3% |
| ResNet-18 crop | 69.9% | 59.4% | 56.5% | 51.9% | 50.6% | 54.9% | 68.0% | 65.7% | 57.2% | 81.6% |
| DINOv2 MLP | 85.3% | 74.7% | 81.5% | 74.1% | 61.6% | 68.1% | **83.3%** | 82.5% | 74.2% | 89.9% |
| ensemble: ResNet crop + DINOv2 | 81.3% | 70.1% | 71.0% | 59.2% | 54.9% | 63.9% | 79.5% | 77.4% | 66.7% | 90.9% |
| ensemble: ResNet squash + DINOv2 | 83.3% | 71.9% | 78.5% | 57.6% | 56.0% | 67.1% | 81.4% | 80.5% | 69.1% | 93.5% |
| router | **87.3%** | **82.4%** | **87.1%** | **79.4%** | **65.2%** | **73.8%** | 81.6% | **83.4%** | 79.2% | |

## AUROC

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 squash224 | 0.803 | 0.685 | 0.800 | 0.529 | 0.531 | 0.684 | 0.781 | 0.800 | 0.672 |
| ResNet-18 crop | 0.769 | 0.630 | 0.641 | 0.509 | 0.489 | 0.632 | 0.750 | 0.715 | 0.612 |
| DINOv2 MLP | **0.927** | **0.854** | 0.901 | **0.823** | **0.669** | 0.750 | **0.911** | **0.916** | 0.821 |
| ensemble: ResNet crop + DINOv2 | 0.910 | 0.816 | 0.860 | 0.751 | 0.639 | 0.759 | 0.895 | 0.890 | 0.789 |
| ensemble: ResNet squash + DINOv2 | 0.922 | 0.845 | **0.922** | 0.755 | 0.655 | **0.788** | 0.908 | 0.915 | 0.814 |

## Class recall (7-generator mean): real / generated

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 squash224 | 86% / 58% | 88% / 39% | 42% / 91% | 11% / 93% | 20% / 85% | 88% / 30% | 86% / 55% | 88% / 55% |
| ResNet-18 crop | 84% / 56% | 87% / 32% | 27% / 86% | 14% / 90% | 15% / 86% | 89% / 20% | 84% / 52% | 83% / 48% |
| DINOv2 MLP | 87% / 83% | 92% / 57% | 82% / 81% | 71% / 77% | 48% / 76% | 79% / 58% | 88% / 79% | 93% / 72% |
| ensemble: ResNet crop + DINOv2 | 90% / 73% | 94% / 46% | 50% / 92% | 26% / 93% | 20% / 90% | 90% / 38% | 90% / 69% | 92% / 63% |
| ensemble: ResNet squash + DINOv2 | 91% / 76% | 94% / 49% | 62% / 95% | 20% / 95% | 22% / 90% | 90% / 44% | 91% / 72% | 93% / 68% |

## Differences in balanced accuracy (pp), 7-generator mean, paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 crop − ResNet-18 squash224 | -2.3* [-3.1, -1.5] | -3.9* [-4.7, -3.2] | -9.9* [-10.8, -9.0] | +0.2 [-0.5, +0.8] | -1.9* [-2.6, -1.2] | -4.2* [-4.9, -3.5] | -2.3* [-3.1, -1.6] | -5.8* [-6.7, -5.0] |
| DINOv2 MLP − ResNet-18 squash224 | +13.1* [+12.0, +14.2] | +11.4* [+10.2, +12.6] | +15.1* [+13.9, +16.4] | +22.4* [+21.2, +23.5] | +9.1* [+7.8, +10.5] | +9.0* [+7.6, +10.4] | +13.0* [+11.8, +14.2] | +11.0* [+9.9, +12.1] |
| DINOv2 MLP − ResNet-18 crop | +15.4* [+14.2, +16.4] | +15.4* [+14.2, +16.6] | +25.0* [+23.7, +26.2] | +22.2* [+20.9, +23.3] | +11.0* [+9.7, +12.4] | +13.2* [+11.9, +14.5] | +15.3* [+14.2, +16.6] | +16.8* [+15.7, +18.1] |
| ensemble: ResNet crop + DINOv2 − DINOv2 MLP | -4.0* [-4.9, -3.2] | -4.6* [-5.6, -3.8] | -10.6* [-11.6, -9.5] | -14.9* [-15.9, -13.9] | -6.7* [-7.8, -5.7] | -4.2* [-5.3, -3.2] | -3.9* [-4.7, -3.0] | -5.1* [-6.0, -4.2] |
| ensemble: ResNet crop + DINOv2 − ResNet-18 crop | +11.3* [+10.4, +12.2] | +10.7* [+9.9, +11.6] | +14.4* [+13.5, +15.4] | +7.3* [+6.5, +8.0] | +4.3* [+3.5, +5.2] | +9.0* [+8.0, +10.0] | +11.5* [+10.6, +12.4] | +11.8* [+10.9, +12.7] |
| ensemble: ResNet squash + DINOv2 − DINOv2 MLP | -2.0* [-2.8, -1.2] | -2.8* [-3.7, -2.0] | -3.0* [-4.0, -2.0] | -16.5* [-17.5, -15.4] | -5.5* [-6.6, -4.5] | -1.0 [-2.1, +0.1] | -2.0* [-2.8, -1.1] | -2.0* [-2.8, -1.1] |
| router − DINOv2 MLP | +2.0* [+1.1, +2.9] | +7.6* [+6.6, +8.7] | +5.6* [+4.6, +6.5] | +5.3* [+4.2, +6.5] | +3.6* [+2.3, +4.9] | +5.7* [+4.6, +7.0] | -1.8* [-2.6, -0.9] | +0.9 [-0.0, +1.9] |
| router − ResNet-18 squash224 | +15.1* [+14.1, +16.1] | +19.1* [+18.0, +20.1] | +20.7* [+19.5, +21.8] | +27.7* [+26.6, +28.8] | +12.8* [+11.4, +14.1] | +14.8* [+13.6, +16.0] | +11.2* [+10.4, +12.1] | +11.9* [+10.9, +13.0] |

## Format dependence (E1): change in balanced accuracy when both classes share the real photos' format (pp)

| Method | Format q96 − clean | Format 256+q96 − clean |
|---|---:|---:|
| ResNet-18 squash224 | -1.9* [-3.2, -0.5] | -0.7 [-1.9, +0.6] |
| ResNet-18 crop | -1.9* [-3.3, -0.8] | -4.3* [-5.6, -3.0] |
| DINOv2 MLP | -2.0* [-3.1, -0.8] | -2.8* [-4.0, -1.7] |
| ensemble: ResNet crop + DINOv2 | -1.8* [-2.9, -0.7] | -3.9* [-4.9, -2.7] |
| ensemble: ResNet squash + DINOv2 | -2.0* [-3.1, -0.9] | -2.8* [-3.8, -1.7] |

## Clean balanced accuracy per held-out generator

| Held-out generator | ResNet-18 squash224 | ResNet-18 crop | DINOv2 MLP | ensemble: ResNet crop + DINOv2 | ensemble: ResNet squash + DINOv2 | router |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | 75.7% | 76.3% | 92.8% | 93.4% | 94.3% | **98.2%** |
| VQDM | 55.3% | 55.1% | 86.1% | 68.9% | 72.9% | **88.5%** |
| SD v1.5 | 79.8% | 78.3% | 85.6% | 87.1% | 87.3% | **95.3%** |
| Wukong | 79.4% | 78.1% | 86.5% | 88.6% | 87.7% | **92.2%** |
| ADM | 61.6% | 55.7% | **76.6%** | 65.9% | 69.5% | 67.1% |
| GLIDE | 90.4% | 89.2% | 91.5% | 91.0% | 94.2% | **94.9%** |
| Midjourney | 63.3% | 56.9% | **78.1%** | 74.0% | 77.5% | 75.0% |

## JPEG Q50 balanced accuracy per held-out generator

| Held-out generator | ResNet-18 squash224 | ResNet-18 crop | DINOv2 MLP | ensemble: ResNet crop + DINOv2 | ensemble: ResNet squash + DINOv2 | router |
|---|---:|---:|---:|---:|---:|---:|
| BigGAN | 51.7% | 47.8% | 84.6% | 71.0% | 72.5% | **90.9%** |
| VQDM | 51.1% | 51.2% | 79.8% | 62.6% | 63.1% | **86.5%** |
| SD v1.5 | 78.2% | 77.4% | 76.8% | 83.6% | 83.0% | **86.7%** |
| Wukong | 78.2% | 76.8% | 76.9% | 80.8% | 82.5% | **88.2%** |
| ADM | 51.0% | 49.1% | 64.2% | 52.3% | 53.9% | **70.6%** |
| GLIDE | 75.3% | 62.1% | 76.7% | 79.0% | **83.6%** | 82.7% |
| Midjourney | 57.6% | 51.1% | 64.2% | 61.4% | 64.8% | **71.0%** |

## Router choice (on degraded seen-generator validation)

| Held-out generator | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---|---|---|---|---|---|---|---|
| BigGAN | DINOv2 L12-24 MLP | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L12-24 MLP | DINOv2 L12-24 LR | ensemble: ResNet squash + DINOv2 | DINOv2 L16 patch MLP |
| VQDM | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 MLP | DINOv2 L20 MLP | ensemble: ResNet squash + DINOv2 | DINOv2 L16 MLP |
| SD v1.5 | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | ensemble: ResNet squash + DINOv2 | DINOv2 L12-24 MLP |
| Wukong | DINOv2 L12 MLP | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | ensemble: ResNet squash + DINOv2 | DINOv2 L16 patch MLP |
| ADM | DINOv2 L12 MLP | DINOv2 L24 + CLIP MLP | DINOv2 L12-24 LR | DINOv2 L12-24 MLP | DINOv2 L24 patch MLP | DINOv2 L12-24 LR | ensemble: ResNet squash + DINOv2 | DINOv2 L12-24 MLP |
| GLIDE | DINOv2 L12 MLP | DINOv2 L16 patch MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | ensemble: ResNet squash + DINOv2 | DINOv2 L16 patch MLP |
| Midjourney | DINOv2 L12-24 LR | DINOv2 L20 MLP | DINOv2 L12-24 LR | DINOv2 L12-24 LR | DINOv2 L24 patch MLP | DINOv2 L24 patch MLP | ensemble: ResNet squash + DINOv2 | DINOv2 L12-24 LR |
