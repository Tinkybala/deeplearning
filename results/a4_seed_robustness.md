# A4: MLP training-seed robustness

The clean+diff MLP was refit with 5 training seeds (42, 1, 2, 3, 4; 42 is the main runs' seed). The seed sets the network initialisation and the 10% early-stopping split; data split, embeddings and the two logistic regressions (deterministic) are unchanged. **fusion** and **router** are the E4 methods rebuilt on each seed's MLP. Cells: mean ± standard deviation over seeds [min, max] of the 7-generator mean.

## Balanced accuracy (%), 7-generator mean

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| logreg_clean (deterministic) | 80.5 | 74.8 | 77.9 | 73.1 | 63.8 | 70.1 |
| logreg_clean+diff (deterministic) | 83.0 | 71.5 | 78.6 | 70.2 | 58.7 | 64.4 |
| mlp_clean+diff | 85.3 ± 0.2 [84.9, 85.5] | 74.5 ± 0.9 [73.3, 75.4] | 81.8 ± 0.2 [81.5, 82.1] | 73.9 ± 0.3 [73.6, 74.2] | 61.5 ± 0.4 [61.2, 62.2] | 67.8 ± 0.4 [67.3, 68.2] |
| fusion | 83.6 ± 0.2 [83.4, 83.9] | 76.5 ± 0.2 [76.2, 76.7] | 80.6 ± 0.3 [80.3, 81.0] | 74.4 ± 0.3 [74.0, 74.7] | 64.1 ± 0.2 [63.8, 64.3] | 70.5 ± 0.2 [70.2, 70.7] |
| router | 85.2 ± 0.2 [85.0, 85.5] | 76.8 ± 0.4 [76.3, 77.1] | 82.0 ± 0.1 [81.9, 82.1] | 75.0 ± 0.2 [74.8, 75.3] | 64.4 ± 0.2 [64.1, 64.6] | 70.9 ± 0.2 [70.5, 71.1] |

## AUROC, 7-generator mean

| Method | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| logreg_clean (deterministic) | 0.883 | 0.827 | 0.864 | 0.811 | 0.697 | 0.774 |
| logreg_clean+diff (deterministic) | 0.905 | 0.804 | 0.869 | 0.779 | 0.621 | 0.691 |
| mlp_clean+diff | 0.927 ± 0.001 [0.926, 0.927] | 0.853 ± 0.004 [0.848, 0.856] | 0.900 ± 0.001 [0.900, 0.901] | 0.823 ± 0.002 [0.820, 0.825] | 0.669 ± 0.003 [0.664, 0.673] | 0.749 ± 0.003 [0.744, 0.751] |
| fusion | 0.913 ± 0.001 [0.912, 0.914] | 0.853 ± 0.001 [0.852, 0.855] | 0.891 ± 0.001 [0.890, 0.892] | 0.833 ± 0.000 [0.832, 0.833] | 0.704 ± 0.001 [0.702, 0.705] | 0.784 ± 0.001 [0.782, 0.785] |

## Key differences in 7-generator mean balanced accuracy (pp): mean ± SD over seeds (seeds agreeing in sign with the mean)

| Comparison | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 |
|---|---:|---:|---:|---:|---:|---:|
| mlp_clean+diff − logreg_clean | +4.7 ± 0.2 (5/5) | -0.3 ± 0.9 (3/5) | +3.9 ± 0.2 (5/5) | +0.8 ± 0.3 (5/5) | -2.3 ± 0.4 (5/5) | -2.3 ± 0.4 (5/5) |
| mlp_clean+diff − logreg_clean+diff | +2.2 ± 0.2 (5/5) | +3.0 ± 0.9 (5/5) | +3.2 ± 0.2 (5/5) | +3.7 ± 0.3 (5/5) | +2.8 ± 0.4 (5/5) | +3.5 ± 0.4 (5/5) |
| router − mlp_clean+diff | -0.0 ± 0.2 (3/5) | +2.2 ± 0.8 (5/5) | +0.2 ± 0.2 (4/5) | +1.2 ± 0.5 (5/5) | +2.9 ± 0.3 (5/5) | +3.1 ± 0.3 (5/5) |
| router − logreg_clean | +4.7 ± 0.2 (5/5) | +1.9 ± 0.4 (5/5) | +4.1 ± 0.1 (5/5) | +2.0 ± 0.2 (5/5) | +0.6 ± 0.2 (5/5) | +0.8 ± 0.2 (5/5) |
| fusion − mlp_clean+diff | -1.7 ± 0.4 (5/5) | +2.0 ± 0.8 (5/5) | -1.2 ± 0.3 (5/5) | +0.5 ± 0.6 (3/5) | +2.6 ± 0.6 (5/5) | +2.7 ± 0.2 (5/5) |

## mlp_clean+diff per held-out generator across seeds

| Held-out generator | Clean BA (%) | Clean AUROC | Resize 32 BA (%) |
|---|---:|---:|---:|
| BigGAN | 92.8 ± 0.4 [92.3, 93.2] | 0.982 ± 0.001 [0.980, 0.983] | 65.4 ± 0.6 [64.5, 66.1] |
| VQDM | 85.8 ± 0.4 [85.3, 86.1] | 0.935 ± 0.002 [0.933, 0.939] | 63.0 ± 0.9 [61.6, 63.7] |
| SD v1.5 | 85.2 ± 0.6 [84.4, 86.0] | 0.927 ± 0.001 [0.926, 0.929] | 59.6 ± 0.6 [58.9, 60.5] |
| Wukong | 86.1 ± 0.3 [85.9, 86.5] | 0.942 ± 0.001 [0.941, 0.942] | 62.3 ± 0.5 [61.8, 63.0] |
| ADM | 76.1 ± 0.5 [75.3, 76.6] | 0.843 ± 0.004 [0.838, 0.848] | 61.0 ± 0.4 [60.3, 61.2] |
| GLIDE | 92.2 ± 0.5 [91.5, 92.9] | 0.981 ± 0.001 [0.979, 0.982] | 62.9 ± 1.0 [61.8, 64.3] |
| Midjourney | 78.7 ± 1.3 [77.0, 80.4] | 0.877 ± 0.002 [0.875, 0.879] | 56.4 ± 0.8 [55.3, 57.5] |
