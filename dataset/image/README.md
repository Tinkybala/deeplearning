# Image data layout

Keep image data under this directory. Large image files are ignored by Git; the folder notes and code remain versioned.

```text
dataset/image/
├── training/
│   ├── GenImage/
│   │   └── <generator>/
│   │       ├── train/{ai,nature}/
│   │       └── val/{ai,nature}/
│   └── CiFake/
│       └── train/{REAL,FAKE}/
└── test/
    ├── GenImage/
    │   └── <held-out-generator>/test/{ai,nature}/
    └── CiFake/
        └── test/{REAL,FAKE}/
```

`training/GenImage` contains the original seven-generator archive and its original `train` and `val` splits. The current held-out GLIDE test images are copied into `test/GenImage/imagenet_glide/test`. Keep this test copy fixed while using the seen-generator validation images for model selection and threshold calibration.

CIFAKE is not present in this workspace yet. Download/obtain it separately, then place its `train/REAL`, `train/FAKE`, `test/REAL`, and `test/FAKE` folders under the matching `CiFake` directories above. The RIGID config and embedding extraction config use these paths.

The ResNet script accepts explicit `--data-root` and `--test-data-root` values. For another GenImage holdout, populate the corresponding folder in `test/GenImage/<generator>/test/{ai,nature}` or pass another test root with the same layout.
