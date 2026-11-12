# Frozen ResNet weights

- `torch_hub/checkpoints/resnet18-f37072fd.pth` is the ImageNet-pretrained starting point for ResNet-18.
- `checkpoints/<run-name>/best_model.pt` contains each fine-tuned leave-one-generator-out checkpoint.

The training script saves and loads its checkpoints here. The pretrained weights are cached here by `torch.hub` as well. Binary weights are ignored by Git because the run checkpoints are large; the JSON summaries, CSV predictions, and training histories remain under `results/`.
