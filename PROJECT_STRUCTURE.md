# Project file and folder guide

This guide explains the cleaned local layout and the purpose of the main code and result files.

## Directory overview

```text
deeplearning-michael/
├── model/
│   ├── ResNet/Frozen Resnet model/
│   │   ├── torch_hub/checkpoints/resnet18-f37072fd.pth
│   │   └── checkpoints/<run-name>/best_model.pt
│   └── DINOv2/Frozen DINO model/
│       ├── torch_hub/                 # populated on first RIGID/DINOv2 load
│       └── probes/*.joblib
├── dataset/image/
│   ├── training/
│   │   ├── GenImage/<generator>/{train,val}/{ai,nature}/
│   │   └── CiFake/train/{REAL,FAKE}/
│   └── test/
│       ├── GenImage/imagenet_glide/test/{ai,nature}/
│       └── CiFake/test/{REAL,FAKE}/
├── results/
│   ├── resnet_holdout_imagenet_glide/
│   ├── resnet_jpeg_holdout_imagenet_glide/
│   ├── resnet_resize_holdout_imagenet_glide/
│   └── resnet_robust_holdout_imagenet_glide/
├── reports/figures/
├── scripts/
├── configs/
├── utils/
├── rigid.py
├── rigid_probe.py
└── project/documentation files
```

The folders named `GenImage` and `CiFake` are inside the `image/training` or `image/test` split exactly as shown. GenImage retains its original `train`/`val` folders under each generator. The current GLIDE validation examples used for final testing were copied into the separate `test/GenImage` tree. CIFAKE was not found locally; its directories are currently empty and ready for the dataset files.

The `.gitignore` excludes image data, local environments, downloaded DINOv2 cache files, and large model weights. It keeps the README files, source code, result CSV/JSON files, training histories, and graphs visible for a GitHub upload.

## Model files

- **`model/ResNet/Frozen Resnet model/torch_hub/checkpoints/resnet18-f37072fd.pth`** — ImageNet-pretrained starting weights used by the ResNet-18 training script.
- **`model/ResNet/Frozen Resnet model/checkpoints/<run-name>/best_model.pt`** — each fine-tuned ResNet checkpoint, selected on seen-generator validation data. Four GLIDE-holdout checkpoints are currently present.
- **`model/DINOv2/Frozen DINO model/torch_hub/`** — location where `rigid.py` caches the DINOv2 source and frozen ViT-L/14 weights on first use. The backbone weights had not been downloaded into this workspace at the time this layout was made.
- **`model/DINOv2/Frozen DINO model/probes/`** — saved logistic-regression and MLP probes built on DINOv2 clean/noisy embedding features. Their metadata lists ImageNet and Synthbuster as training sources; no CIFAKE-trained probe artifact is currently in this folder. These are learned classifiers on top of a frozen backbone; the plain RIGID similarity score itself is training-free.

## Result files

Each `results/resnet_*_holdout_imagenet_glide/` folder contains:

- **`summary.json`** — split sizes, best epoch, selected threshold, device/software, and metrics for all six test conditions.
- **`training_history.json`** — per-epoch fit accuracy/loss and seen-generator validation balanced accuracy.
- **`test_scores.csv`** — per-image label, real-image score, prediction, condition, and a portable path to the test image.

The `.pt` checkpoint for a run is stored in the `model/ResNet` hierarchy, not beside its result CSV. This keeps model binaries together and leaves result tables small enough to publish directly.

## Python code

### Main detector code

- **`rigid.py`** — frozen DINOv2 model loader and the RIGID score: cosine similarity between the clean image embedding and an embedding after normalized-pixel Gaussian noise. Higher similarity means more real-like. The DINOv2 cache is directed into the requested model tree.
- **`rigid_probe.py`** — builds clean, noisy, difference, and cosine features; loads a saved probe; predicts a real/fake probability for an image.
- **`scripts/train_resnet_generalization.py`** — trains ImageNet-pretrained ResNet-18 with a selectable held-out GenImage generator; calibrates its threshold on seen-generator validation data; evaluates clean, JPEG, resize, and combined test conditions. Defaults now point to `dataset/image/training/GenImage` and `dataset/image/test/GenImage`; checkpoints go under `model/ResNet`.
- **`scripts/evaluate.py`** — runs the training-free RIGID score on a dataset described by a YAML config and writes per-image scores plus summary metrics. Relative paths resolve from the project root.

### RIGID feature and probe code

- **`scripts/extract_embeddings.py`** — saves DINOv2 clean/noisy embeddings and RIGID cosine scores for the sets listed in `configs/embeddings.yaml`.
- **`scripts/train_probe.py`** — compares feature sets with logistic-regression or MLP probes under within-dataset, leave-one-generator-out, and cross-dataset protocols. Its CIFAKE within-dataset split uses `train` for fitting and `test` for scoring.
- **`scripts/train_final_probe.py`** — fits one selected probe on the cached embeddings and saves it under `model/DINOv2/Frozen DINO model/probes/` by default.
- **`scripts/diagnose.py`** — compares RIGID score distributions across multiple image sources and preprocessing pipelines, and saves pairwise AUROC tables and a plot.

### Data preparation and reporting utilities

- **`scripts/make_report_assets.py`** — converts existing ResNet score paths to the new dataset tree and regenerates the report graphs.
- **`scripts/download_raise1k.py`** — downloads the RAISE-1k real-image dataset from its official package using a supplied CSV.
- **`scripts/fetch_zip_subset.py`** — retrieves a random sample from a remote ZIP using HTTP range requests rather than downloading the whole archive.
- **`utils/datasets.py`** — dataset registry, folder scanning, image loading, and balanced per-class sampling. CIFAKE now maps to the local `dataset/image/training/CiFake` and `dataset/image/test/CiFake` folders.
- **`utils/sources.py`** — local-folder and Kaggle dataset fetch helpers.
- **`utils/__init__.py`** — exposes the registry and common dataset-loading functions.

## Configs and notebooks

- **`configs/cifake.yaml`** — RIGID evaluation on up to 1,000 images per class from CIFAKE's test split; it uses the fixed threshold and does not train a detector. The separate `scripts/train_probe.py` workflow can train a DINOv2 feature probe on CIFAKE's training split and score its test split after embeddings are extracted.
- **`configs/embeddings.yaml`** — image sources used for DINOv2 feature extraction, including the new local CIFAKE train/test paths. Other legacy ImageNet/Synthbuster paths in this file still need to be set to the local machine's data locations before those sets can be run.
- **`configs/diagnose.yaml`** — data sets and preprocessing recipes for RIGID diagnostics; CIFAKE paths use the new tree, while unrelated legacy sets retain their own external paths.
- **`configs/rigid_imagenet.yaml`** — config for the separate RIGID ImageNet256 benchmark; its external dataset root must be configured before use.
- **`configs/synthbuster.yaml`** — config for Synthbuster fakes versus RAISE-1k photos; those datasets are separate from the GenImage and CIFAKE tree.
- **`RIGID_single_image.ipynb`** — interactive example scoring a single image with training-free RIGID.
- **`RIGID_probe_single_image.ipynb`** — interactive example using a saved DINOv2 feature probe; its probe paths have been updated.
- **`anomaly_detection.ipynb`** — exploratory analysis of embedding changes and real-only anomaly detectors, with saved probe paths updated.

## Project metadata and reports

- **`pyproject.toml` / `uv.lock`** — project dependencies and locked environment.
- **`README.md`** — GitHub landing page with current results and reproducibility commands.
- **`REPORT.md`** — detailed ongoing ResNet study with graphs, metrics, methods, and limitations.
- **`RESNET_EXPERIMENT.md`** — compact experiment notes.
- **`reports/figures/`** — balanced accuracy, AUROC, and training-curve PNGs.
- **`dataset/image/README.md` and the CiFake/test README files** — expected data placement and notes on which image files are absent.

### Paths still specific to other experiments

The active ResNet and CIFAKE/RIGID paths now use the organized project tree. Configs for separate Synthbuster, RAISE-1k, and paper ImageNet256 experiments intentionally remain separate and include machine-specific external data paths that need editing before those unrelated experiments are run.
