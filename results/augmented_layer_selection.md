# Augmented training for every DINOv2 layer probe, with nested selection

32 single detectors: 8 DINOv2 feature sets × {LR, MLP} × {clean, clean + augmented} training, each with the clean seen-generator validation threshold (single images, no condition knowledge). **Inner** = nested leave-one-generator-out balanced accuracy (refit on five seen generators, scored on the sixth), mean over the six standard conditions and the six inner folds, averaged over the 7 outer folds; it never uses the outer test generator. **Outer** = test balanced accuracy on the held-out generator. **DINOv2 L20 MLP [clean+both]** is the final detector of final_detector_experiments.py. Mean over the 7 holdouts.

## All candidates, sorted by inner score

| Candidate | Inner (6 cond.) | Seen val | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Outer mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 L12-24 LR [clean+both] | 81.3% | 96.0% | 89.3% | 83.8% | 87.5% | 82.3% | 73.8% | 76.7% | 86.1% | 82.2% |
| **DINOv2 L20 MLP [clean+both]** | 80.6% | 95.1% | 88.7% | 83.2% | 86.7% | 80.7% | 72.3% | 76.4% | 85.5% | 81.3% |
| DINOv2 L12-24 MLP [clean+both] | 79.3% | 95.9% | 88.5% | 80.8% | 87.3% | 80.8% | 72.7% | 74.6% | 83.9% | 80.8% |
| DINOv2 L20 LR [clean+both] | 78.7% | 93.5% | 86.5% | 81.5% | 84.6% | 78.8% | 69.9% | 74.8% | 83.6% | 79.3% |
| DINOv2 L16 patch MLP [clean+both] | 78.3% | 95.5% | 86.9% | 81.7% | 85.3% | 79.0% | 69.2% | 74.5% | 84.9% | 79.4% |
| DINOv2 L24 patch MLP [clean+both] | 77.9% | 89.8% | 84.5% | 81.2% | 82.6% | 78.9% | 69.9% | 76.9% | 83.8% | 79.0% |
| DINOv2 L16 patch LR [clean+both] | 76.4% | 93.4% | 84.4% | 79.4% | 82.6% | 77.0% | 66.8% | 72.5% | 82.2% | 77.1% |
| DINOv2 L24 patch LR [clean+both] | 76.3% | 87.6% | 82.7% | 78.7% | 80.6% | 76.0% | 67.5% | 73.5% | 82.0% | 76.5% |
| DINOv2 L24 MLP [clean+both] | 75.8% | 86.4% | 82.1% | 77.8% | 79.6% | 75.7% | 68.0% | 72.6% | 81.2% | 76.0% |
| DINOv2 L24 patch MLP [clean] | 75.8% | 87.7% | 83.1% | 79.7% | 81.2% | 75.7% | 62.9% | 74.7% | 82.2% | 76.2% |
| DINOv2 L20 MLP [clean] | 75.6% | 94.6% | 88.6% | 74.7% | 85.8% | 78.5% | 66.4% | 66.3% | 83.4% | 76.7% |
| DINOv2 L24 patch LR [clean] | 75.4% | 88.2% | 83.0% | 79.1% | 80.8% | 75.8% | 63.6% | 73.7% | 82.2% | 76.0% |
| DINOv2 L24 MLP [clean] | 74.6% | 85.5% | 81.5% | 76.9% | 79.1% | 74.4% | 66.1% | 72.4% | 80.3% | 75.1% |
| DINOv2 L20 LR [clean] | 74.6% | 94.0% | 87.6% | 74.1% | 85.2% | 77.1% | 64.3% | 66.0% | 84.1% | 75.7% |
| DINOv2 L16 LR [clean+both] | 74.4% | 95.0% | 86.7% | 74.3% | 83.2% | 75.0% | 66.4% | 68.5% | 81.5% | 75.7% |
| DINOv2 L16 patch MLP [clean] | 73.9% | 95.6% | 86.6% | 76.0% | 84.2% | 77.4% | 60.5% | 64.4% | 84.0% | 74.9% |
| DINOv2 L24 LR [clean+both] | 73.9% | 84.2% | 80.9% | 76.5% | 79.0% | 74.4% | 66.6% | 71.2% | 79.7% | 74.8% |
| DINOv2 L16 MLP [clean+both] | 72.9% | 96.9% | 84.5% | 69.7% | 82.9% | 74.6% | 67.4% | 65.7% | 77.8% | 74.1% |
| DINOv2 L24 LR [clean] | 72.7% | 84.8% | 80.5% | 74.8% | 77.9% | 73.1% | 63.8% | 70.1% | 78.9% | 73.4% |
| DINOv2 L16 patch LR [clean] | 71.5% | 93.8% | 85.0% | 72.1% | 82.6% | 75.0% | 55.8% | 60.8% | 82.1% | 71.9% |
| DINOv2 L12-24 LR [clean] | 69.2% | 97.3% | 88.5% | 66.7% | 84.7% | 72.0% | 53.3% | 56.0% | 83.2% | 70.2% |
| DINOv2 L12 LR [clean+both] | 68.3% | 94.2% | 80.1% | 64.8% | 76.4% | 69.5% | 64.6% | 55.5% | 69.2% | 68.5% |
| DINOv2 L12-24 MLP [clean] | 68.1% | 97.1% | 88.3% | 60.3% | 84.3% | 66.5% | 51.1% | 52.3% | 82.0% | 67.1% |
| DINOv2 L12 MLP [clean+both] | 67.1% | 96.6% | 80.6% | 64.8% | 75.7% | 69.0% | 65.7% | 55.1% | 67.7% | 68.5% |
| DINOv2 L8 MLP [clean+both] | 63.3% | 89.3% | 73.5% | 63.8% | 70.7% | 63.7% | 60.0% | 54.3% | 62.6% | 64.3% |
| DINOv2 L8 LR [clean+both] | 63.0% | 85.7% | 72.0% | 63.3% | 72.1% | 62.3% | 59.1% | 52.2% | 63.7% | 63.5% |
| DINOv2 L16 MLP [clean] | 62.9% | 96.7% | 85.3% | 56.6% | 78.6% | 55.9% | 50.9% | 50.7% | 76.9% | 63.0% |
| DINOv2 L16 LR [clean] | 62.3% | 95.8% | 86.5% | 62.3% | 68.5% | 54.0% | 49.8% | 53.8% | 80.3% | 62.5% |
| DINOv2 L12 MLP [clean] | 59.6% | 97.0% | 83.6% | 53.1% | 65.4% | 52.8% | 50.0% | 50.0% | 67.5% | 59.2% |
| DINOv2 L12 LR [clean] | 58.7% | 95.0% | 82.7% | 54.0% | 63.3% | 52.9% | 50.1% | 50.3% | 70.7% | 58.9% |
| DINOv2 L8 MLP [clean] | 56.1% | 89.9% | 78.1% | 59.1% | 51.5% | 52.0% | 50.0% | 51.1% | 64.2% | 57.0% |
| DINOv2 L8 LR [clean] | 56.0% | 87.2% | 75.4% | 56.7% | 52.2% | 50.4% | 50.3% | 50.8% | 66.1% | 56.0% |

## Selected detectors

| Selection rule | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 L20 MLP [clean+both] | 88.7% | 83.2% | 86.7% | 80.7% | 72.3% | 76.4% | 85.5% | 81.3% |
| nested pick (mean of 6 conditions) | 88.6% | 82.8% | 86.5% | 80.3% | 72.5% | 76.3% | 85.4% | 81.2% |
| nested pick (clean) | 86.9% | 74.4% | 85.8% | 77.7% | 65.1% | 68.3% | 83.1% | 76.4% |
| seen-val pick | 87.8% | 67.2% | 84.4% | 70.5% | 55.3% | 56.1% | 82.7% | 70.2% |

## Difference from the final detector (pp), paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format 256+q96 | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nested pick (mean of 6 conditions) − final | -0.2 [-0.7, +0.3] | -0.4 [-1.1, +0.2] | -0.2 [-0.8, +0.4] | -0.4 [-1.1, +0.4] | +0.2 [-0.7, +1.1] | -0.1 [-0.9, +0.8] | -0.1 [-0.8, +0.4] | -0.2 [-0.5, +0.1] |
| nested pick (clean) − final | -1.9* [-2.4, -1.3] | -8.8* [-9.4, -8.2] | -0.9* [-1.5, -0.3] | -3.0* [-3.7, -2.2] | -7.1* [-7.9, -6.3] | -8.1* [-8.8, -7.3] | -2.5* [-3.1, -1.9] | -5.0* [-5.2, -4.7] |
| seen-val pick − final | -1.0* [-1.7, -0.3] | -16.0* [-16.8, -15.1] | -2.3* [-3.1, -1.4] | -10.2* [-11.2, -9.2] | -17.0* [-18.0, -15.8] | -20.3* [-21.2, -19.3] | -2.8* [-3.6, -2.1] | -11.1* [-11.5, -10.7] |

## Picks per outer holdout

| Held-out generator | nested pick (mean of 6 conditions) | nested pick (clean) | seen-val pick |
|---|---|---|---|
| BigGAN | DINOv2 L12-24 LR [clean+both] | DINOv2 L12-24 LR [clean] | DINOv2 L12-24 MLP [clean] |
| VQDM | DINOv2 L20 MLP [clean+both] | DINOv2 L12-24 MLP [clean] | DINOv2 L12-24 LR [clean] |
| SD v1.5 | DINOv2 L12-24 LR [clean+both] | DINOv2 L12-24 LR [clean+both] | DINOv2 L12-24 LR [clean] |
| Wukong | DINOv2 L12-24 LR [clean+both] | DINOv2 L20 MLP [clean+both] | DINOv2 L12-24 LR [clean] |
| ADM | DINOv2 L20 MLP [clean+both] | DINOv2 L20 MLP [clean+both] | DINOv2 L12-24 MLP [clean] |
| GLIDE | DINOv2 L12-24 LR [clean+both] | DINOv2 L20 MLP [clean+both] | DINOv2 L16 MLP [clean+both] |
| Midjourney | DINOv2 L12-24 LR [clean+both] | DINOv2 L12-24 LR [clean+both] | DINOv2 L12-24 LR [clean] |

Nested picks (mean of 6 conditions): DINOv2 L12-24 LR [clean+both] ×5, DINOv2 L20 MLP [clean+both] ×2.
