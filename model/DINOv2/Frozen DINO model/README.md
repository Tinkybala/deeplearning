# Frozen DINOv2 weights and probes

`rigid.py` loads DINOv2 ViT-L/14 through `torch.hub` and freezes the backbone. The first run downloads the DINOv2 source and checkpoint into `torch_hub/` beneath this directory. Those binary/source cache files are ignored by Git.

`probes/` contains the saved logistic-regression and MLP probes trained on DINOv2-derived features. Their metadata lists ImageNet and Synthbuster as training sources; no CIFAKE-trained probe is currently stored here. They are learned classifiers on top of the frozen vision backbone; the base RIGID similarity score itself is training-free.
