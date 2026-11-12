# ResNet-18 generator holdout experiment

For the complete ongoing study, including training curves, test-result graphs,
the JPEG/resize ablation, detailed metrics, and open next steps, see
[`REPORT.md`](REPORT.md). The repository-facing summary is in [`README.md`](README.md).

This experiment fine-tunes ImageNet-pretrained ResNet-18 weights to distinguish
real ImageNet photos from AI-generated images. `imagenet_glide` is held out
completely; the other six generators are used for training and validation.

## Data split

- **Fit:** generated train images from the six seen generators and 80% of the
  unique real train images.
- **Threshold/model selection:** validation images from the six seen generators
  plus the reserved 20% of real train images. The shared ImageNet validation
  photos are not used here because they also appear in the held-out folders.
- **Final test:** all 500 real and 500 fake images at
  `dataset/image/test/GenImage/imagenet_glide/test/{nature,ai}`.
- **Robustness:** JPEG quality 50, shorter-side resize to 128/64/32 pixels, and
  128-pixel resize followed by JPEG quality 50. Resizing preserves aspect ratio.
  The threshold stays fixed from the non-held-out calibration set.

## Run on Windows with an NVIDIA GPU

The general project dependency lock installs CPU PyTorch wheels from PyPI.
For the CUDA 13.2 build used in this experiment, set up the project environment
with Python 3.12, then install the official CUDA wheels:

```powershell
uv sync --locked --python C:\Path\To\Python312\python.exe --no-dev
uv pip install --reinstall --python .venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu132 `
  torch==2.15.0+cu132 torchvision==0.29.2+cu132
```

From this directory, train and evaluate with:

```powershell
.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32
```

Use `--eval-only` to repeat held-out evaluation from the saved checkpoint. Use
`--held-out` with another listed generator to run a different holdout experiment.

## Robust augmentation comparison and ablation

Run the augmented variant with the same seed, split, and training settings:

```powershell
.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32 --robust-augmentation
```

For each training image, JPEG compression (quality 30-95) and downsampling to
32, 64, or 128 pixels are independently applied with probability 0.5. Validation
and test preprocessing are unchanged. The saved GLIDE comparison was:

| Condition | Standard balanced accuracy | JPEG + resize balanced accuracy | Change |
|---|---:|---:|---:|
| Clean | 85.1% | 73.2% | -11.9 pp |
| JPEG quality 50 | 56.7% | 67.8% | +11.1 pp |
| Resize to 128 px | 70.8% | 71.2% | +0.4 pp |
| Resize to 64 px | 51.0% | 67.5% | +16.5 pp |
| Resize to 32 px | 50.3% | 63.3% | +13.0 pp |
| Resize to 128 px + JPEG 50 | 61.8% | 61.7% | -0.1 pp |

Each model's threshold was calibrated separately using non-GLIDE data and then
held fixed for all of its test conditions. The augmentation helps on JPEG and
low-resolution GLIDE images, while reducing clean-image performance. To
identify which component is responsible, we trained two more models with only
JPEG compression augmentation or only downsampling augmentation. These used
the same split, seed, number of epochs, batch size, and evaluation protocol.

```powershell
.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32 --augmentation jpeg

.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32 --augmentation resize
```

### Ablation results: balanced accuracy

| GLIDE test condition | No augmentation | JPEG only | Resize only | JPEG + resize |
|---|---:|---:|---:|---:|
| Clean | 85.1% | 84.3% | 75.7% | 73.2% |
| JPEG quality 50 | 56.7% | **76.0%** | 56.6% | 67.8% |
| Resize to 128 px | **70.8%** | 60.3% | 73.8% | 71.2% |
| Resize to 64 px | 51.0% | 51.0% | 66.4% | **67.5%** |
| Resize to 32 px | 50.3% | 50.2% | 61.7% | **63.3%** |
| Resize to 128 px + JPEG 50 | **61.8%** | 56.3% | 58.8% | 61.7% |

AUROC, which does not depend on the selected classification threshold:

| GLIDE test condition | No augmentation | JPEG only | Resize only | JPEG + resize |
|---|---:|---:|---:|---:|
| Clean | 0.931 | 0.930 | 0.836 | 0.806 |
| JPEG quality 50 | 0.572 | **0.843** | 0.608 | 0.745 |
| Resize to 128 px | **0.873** | 0.775 | 0.815 | 0.781 |
| Resize to 64 px | 0.539 | 0.542 | 0.731 | **0.731** |
| Resize to 32 px | 0.575 | 0.546 | 0.680 | **0.687** |
| Resize to 128 px + JPEG 50 | 0.675 | 0.620 | **0.704** | 0.666 |

The ablation gives a clearer, distortion-specific result. JPEG-only training
recovers most of the JPEG-Q50 performance (AUROC 0.572 to 0.843) while retaining
clean AUROC (0.931); it does not improve low-resolution detection. Resize-only
training improves 64/32-pixel detection (balanced accuracy 51.0%/50.3% to
66.4%/61.7%) but lowers clean balanced accuracy by 9.4 percentage points. Using
both components improves low-resolution scores a little further, but does not
combine their benefits on every condition: JPEG-only is substantially better on
JPEG-Q50, and clean performance is lowest with both augmentations. This points
to a practical follow-up: compare JPEG-only against a targeted resize policy,
and report clean performance alongside each robustness result.

These results are from one held-out generator (GLIDE) and one seed, so they are
evidence about this setup rather than a claim that the same effects hold for all
generators. Each model's threshold was calibrated separately on seen-generator
validation data and kept fixed across that model's test transformations; AUROC
is included to show ranking quality independently of that threshold.

## Outputs

Results are written under `results/` in separate folders for each generator and
augmentation setting (for example, `resnet_jpeg_holdout_imagenet_glide/`):

- The best checkpoint and calibrated threshold are in
  `model/ResNet/Frozen Resnet model/checkpoints/<run-name>/best_model.pt`.
- `training_history.json`: per-epoch training and validation metrics
- `test_scores.csv`: per-image scores for each test condition
- `summary.json`: split, runtime, software/device versions, and aggregate metrics

The summary records the trained model, split sizes, chosen threshold, and held-out
metrics. Report the pretrained initialization and the one-generator holdout when
describing these results.
