# SC4001: AI-Generated Image Detection

## Current work

The current experiment fine-tunes an ImageNet-pretrained ResNet-18 to distinguish real ImageNet photographs from AI-generated images. It tests whether the model transfers to a generator it did not see during training: **GLIDE is held out entirely**, and six other generator categories are used to fit the model and select its decision threshold.

The first study compares a standard training setup with artifact augmentations. Follow-up ablations train with JPEG augmentation alone and downsampling augmentation alone. The experiment varies training augmentation while keeping the held-out generator, split, seed, training duration, and test conditions fixed.

A teammate's CIFAKE model was mentioned as parallel project work. Its configuration and result files are not included in this repository snapshot, so the tables below cover only the ImageNet-AI ResNet runs documented here.

### Current held-out results

Balanced accuracy on 500 real and 500 GLIDE-generated images per test condition:

| Test condition | Baseline | JPEG only | Resize only | JPEG + resize |
|---|---:|---:|---:|---:|
| Clean | 85.1% | 84.3% | 75.7% | 73.2% |
| JPEG quality 50 | 56.7% | **76.0%** | 56.6% | 67.8% |
| Resize to 128 px | 70.8% | 60.3% | 73.8% | 71.2% |
| Resize to 64 px | 51.0% | 51.0% | 66.4% | **67.5%** |
| Resize to 32 px | 50.3% | 50.2% | 61.7% | **63.3%** |
| Resize to 128 px + JPEG 50 | **61.8%** | 56.3% | 58.8% | 61.7% |

![Balanced accuracy by model and test condition](reports/figures/test_balanced_accuracy.png)

AUROC is included because it measures score ranking without relying on a single classification threshold:

| Test condition | Baseline | JPEG only | Resize only | JPEG + resize |
|---|---:|---:|---:|---:|
| Clean | 0.931 | 0.930 | 0.836 | 0.806 |
| JPEG quality 50 | 0.572 | **0.843** | 0.608 | 0.745 |
| Resize to 128 px | **0.873** | 0.775 | 0.815 | 0.781 |
| Resize to 64 px | 0.539 | 0.542 | 0.731 | 0.731 |
| Resize to 32 px | 0.575 | 0.546 | 0.680 | 0.687 |
| Resize to 128 px + JPEG 50 | 0.675 | 0.620 | **0.704** | 0.666 |

![AUROC by model and test condition](reports/figures/test_auroc.png)

**Early interpretation:** augmentation effects are distortion-specific. JPEG-only training markedly improves JPEG-Q50 detection while preserving clean AUROC. Resize augmentation improves detection after stronger downsampling, with a clean-image performance cost. These are preliminary results from one held-out generator and one seed, not a conclusion about all generators.

For the full protocol, detailed metrics, training curves, experiment history, limitations, and follow-up questions, see the [detailed report](REPORT.md). The shorter [experiment notes](RESNET_EXPERIMENT.md) document the current setup and commands.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the cleaned folder hierarchy and a guide to the model, dataset, script, config, and notebook files.

## Dataset and split

GenImage is now stored at `dataset/image/training/GenImage`, preserving the original seven generator folders and their `train`/`val` splits. The current held-out GLIDE test copy is at `dataset/image/test/GenImage/imagenet_glide/test`. CIFAKE is not present locally yet; its expected train/test folders are documented in `dataset/image/README.md`.

```text
dataset/image/
├── training/
│   ├── GenImage/<generator>/{train,val}/{ai,nature}/
│   └── CiFake/train/{REAL,FAKE}/
└── test/
    ├── GenImage/<held-out-generator>/test/{ai,nature}/
    └── CiFake/test/{REAL,FAKE}/
```

The ResNet script defaults to the GenImage training/validation root and the separate GenImage test root shown above. For another generator holdout, place its test images under the matching `test/GenImage/<generator>/test/` directory or provide `--test-data-root` and `--test-split` explicitly. CIFAKE is configured as a local dataset under `dataset/image`; place its original train/test category folders at the paths above before running RIGID evaluation or embedding extraction.

Seven generator folders are available:

- `imagenet_ai_0419_biggan`
- `imagenet_ai_0419_vqdm`
- `imagenet_ai_0424_sdv5`
- `imagenet_ai_0424_wukong`
- `imagenet_ai_0508_adm`
- `imagenet_glide` (held out for this experiment)
- `imagenet_midjourney`

Each contains `train` and `val` splits, with `ai` and `nature` image categories. The ImageNet real photographs are shared between generator folders; the experiment counts them once.

For the GLIDE holdout:

- **Fit:** 13,600 images: 12,000 generated images from the six seen generators and 1,600 real train images.
- **Validation and threshold selection:** 3,400 images: 3,000 generated validation images from the seen generators and 400 real train images reserved from the real training photos.
- **Final test:** 1,000 GLIDE validation images: 500 real and 500 generated.

The held-out generator is excluded from both training and threshold/model selection. Real validation photos were reserved from the training split because the validation photos are shared across the generator directories; this prevents overlap between calibration and the held-out test images.

## Reproducing the experiment

The training script uses the new `dataset/image` layout by default. It uses Python 3.12, PyTorch, torchvision, Pillow, scikit-learn, NumPy, and tqdm. The project dependencies are listed in `pyproject.toml`; the recorded runs used CUDA-enabled PyTorch 2.15.0+cu132 and torchvision 0.29.2+cu132 on an NVIDIA RTX 4050 Laptop GPU.

From the repository directory on Windows, after setting up the environment and dataset:

```powershell
.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32 --workers 4

.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32 --workers 4 `
  --augmentation jpeg

.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32 --workers 4 `
  --augmentation resize

.venv\Scripts\python.exe scripts\train_resnet_generalization.py `
  --held-out imagenet_glide --epochs 8 --batch-size 32 --workers 4 `
  --augmentation both
```

The legacy `--robust-augmentation` option is an alias for `--augmentation both`. Choose another generator with `--held-out <generator-folder>` to start a new leave-one-generator-out run. Use `--eval-only` to score an existing local checkpoint again.

After new runs, regenerate the figures and portable result summaries from the four documented GLIDE runs with:

```powershell
.venv\Scripts\python.exe scripts\make_report_assets.py
```

Once CIFAKE has been placed under `dataset/image/training/CiFake` and `dataset/image/test/CiFake`, run the frozen DINOv2/RIGID test evaluation with:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --config configs\cifake.yaml
```

The base DINOv2 weights are fetched by `torch.hub` on first use and cached under `model/DINOv2/Frozen DINO model/torch_hub/`. This CIFAKE config scores up to 1,000 images per class from the test split; it does not train RIGID or a probe.

Update the README and report tables when adding a new holdout or experiment. Result summaries, per-image scores, and training histories are small text files and can be published to GitHub. Model checkpoints are excluded by `.gitignore` because each is about 43 MB; the image dataset and local virtual environment should also remain outside the repository. The score CSV paths are dataset-relative and do not expose local machine paths.

## Repository map

- `scripts/train_resnet_generalization.py` — training, validation, calibration, held-out evaluation, and score export.
- `scripts/make_report_assets.py` — sanitizes result paths and regenerates the plotted figures.
- `REPORT.md` — detailed, ongoing experiment report.
- `PROJECT_STRUCTURE.md` — folder tree and guide to the code, configs, notebooks, and result files.
- `RESNET_EXPERIMENT.md` — concise experimental setup and ablation notes.
- `results/` — per-run summaries, per-image scores, and training histories; local checkpoints are ignored.
- `model/ResNet/Frozen Resnet model/` — ImageNet initialization and fine-tuned ResNet checkpoints.
- `model/DINOv2/Frozen DINO model/` — cached DINOv2 backbone and saved DINOv2 feature probes.
- `dataset/image/` — organized GenImage and CIFAKE image data.

The model folder layout is:

```text
model/
├── ResNet/Frozen Resnet model/
│   ├── torch_hub/checkpoints/resnet18-f37072fd.pth
│   └── checkpoints/<run-name>/best_model.pt
└── DINOv2/Frozen DINO model/
    ├── torch_hub/        # downloaded DINOv2 source and frozen weights
    └── probes/           # saved logistic-regression and MLP probes
```

The DINOv2 backbone cache is populated on first use. It is not currently stored in the workspace; its path is now fixed under the model folder.

## Ongoing work

This is a working research record, not a finished claim. Potential next steps include repeating the comparison with multiple random seeds, rotating the held-out generator, exploring less costly resize augmentation, and inspecting errors by generator and image category. Add future runs to the report with the holdout, seed, split, training settings, threshold calibration method, and all tested conditions so comparisons remain interpretable.
