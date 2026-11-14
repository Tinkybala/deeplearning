# E2: format-aligned training

**clean-trained** = the main runs' probes. **aligned-trained** = the same probes refit on seen-generator images whose real *and* generated images went through Format 256+q96 (resize shorter side to 256, JPEG q96 4:4:4), threshold from the seen-generator validation split under the same pipeline. Both are tested on all eight conditions of the held-out generator (the six standard ones are the unaligned files, exactly as in the main runs). RIGID has no trained part; only its threshold changes. Mean over the 7 held-out generators, worst generator in brackets.

## Balanced accuracy

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rigid (clean-trained) | 75.9% [64.9] | 69.1% [57.2] | 70.3% [56.7] | 57.3% [49.5] | 50.0% [49.6] | 54.4% [47.4] | 73.5% [61.6] | 72.5% [55.3] |
| logreg_clean (clean-trained) | 80.5% [68.6] | 74.8% [62.4] | 77.9% [66.8] | 73.1% [64.7] | 63.8% [58.9] | 70.1% [56.1] | 78.5% [67.5] | 78.9% [63.7] |
| logreg_clean+diff (clean-trained) | 83.0% [74.4] | 71.5% [59.7] | 78.6% [70.4] | 70.2% [63.5] | 58.7% [53.7] | 64.4% [53.6] | 80.2% [69.4] | 79.1% [64.0] |
| mlp_clean+diff (clean-trained) | 85.3% [76.6] | 74.7% [64.2] | 81.5% [73.7] | 74.1% [68.2] | 61.6% [56.5] | 68.1% [56.5] | 83.3% [71.6] | 82.5% [68.6] |
| rigid (aligned-trained) | 73.2% [65.4] | 64.5% [57.6] | 65.4% [55.7] | 53.0% [50.2] | 50.0% [49.7] | 50.8% [47.0] | 71.7% [63.6] | 72.4% [56.8] |
| logreg_clean (aligned-trained) | 80.2% [71.2] | 74.3% [66.5] | 76.7% [69.1] | 71.1% [65.3] | 62.6% [58.9] | 70.5% [59.9] | 79.1% [68.4] | 80.1% [68.1] |
| logreg_clean+diff (aligned-trained) | 81.2% [73.7] | 72.4% [60.6] | 75.3% [71.4] | 68.1% [65.1] | 58.0% [53.7] | 63.6% [53.9] | 79.9% [70.9] | 83.3% [73.8] |
| mlp_clean+diff (aligned-trained) | 83.6% [75.6] | 74.5% [60.8] | 80.0% [75.5] | 72.5% [68.2] | 61.0% [56.6] | 66.8% [56.5] | 82.7% [73.8] | 84.8% [75.6] |

## AUROC

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rigid (clean-trained) | 0.839 [0.717] | 0.770 [0.606] | 0.791 [0.591] | 0.692 [0.484] | 0.574 [0.396] | 0.655 [0.454] | 0.816 [0.694] | 0.825 [0.666] |
| logreg_clean (clean-trained) | 0.883 [0.768] | 0.827 [0.694] | 0.864 [0.742] | 0.811 [0.700] | 0.697 [0.609] | 0.774 [0.622] | 0.867 [0.752] | 0.875 [0.734] |
| logreg_clean+diff (clean-trained) | 0.905 [0.817] | 0.804 [0.694] | 0.869 [0.787] | 0.779 [0.689] | 0.621 [0.529] | 0.691 [0.548] | 0.882 [0.771] | 0.889 [0.770] |
| mlp_clean+diff (clean-trained) | 0.927 [0.844] | 0.854 [0.756] | 0.901 [0.829] | 0.823 [0.736] | 0.669 [0.585] | 0.750 [0.618] | 0.911 [0.807] | 0.916 [0.823] |
| rigid (aligned-trained) | 0.839 [0.717] | 0.770 [0.606] | 0.791 [0.591] | 0.692 [0.484] | 0.574 [0.396] | 0.655 [0.454] | 0.816 [0.694] | 0.825 [0.666] |
| logreg_clean (aligned-trained) | 0.883 [0.778] | 0.827 [0.717] | 0.862 [0.753] | 0.809 [0.707] | 0.696 [0.615] | 0.775 [0.637] | 0.869 [0.752] | 0.882 [0.759] |
| logreg_clean+diff (aligned-trained) | 0.902 [0.800] | 0.801 [0.668] | 0.864 [0.788] | 0.767 [0.698] | 0.619 [0.541] | 0.684 [0.558] | 0.883 [0.766] | 0.908 [0.828] |
| mlp_clean+diff (aligned-trained) | 0.921 [0.829] | 0.840 [0.693] | 0.895 [0.829] | 0.815 [0.742] | 0.665 [0.588] | 0.731 [0.611] | 0.908 [0.803] | 0.927 [0.860] |

## Aligned-trained − clean-trained, balanced accuracy (pp), 7-generator mean, paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rigid | -2.7* [-3.5, -2.1] | -4.6* [-5.3, -3.8] | -4.9* [-5.8, -4.2] | -4.3* [-5.0, -3.6] | -0.0 [-0.2, +0.2] | -3.6* [-4.3, -3.0] | -1.7* [-2.5, -1.0] | -0.1 [-0.7, +0.6] |
| logreg_clean | -0.3 [-1.0, +0.3] | -0.5 [-1.2, +0.2] | -1.2* [-1.9, -0.5] | -1.9* [-2.6, -1.3] | -1.2* [-1.9, -0.5] | +0.3 [-0.4, +1.1] | +0.6 [-0.1, +1.3] | +1.2* [+0.5, +1.8] |
| logreg_clean+diff | -1.8* [-2.6, -1.1] | +0.9* [+0.1, +1.6] | -3.3* [-4.2, -2.5] | -2.1* [-3.0, -1.2] | -0.7 [-1.7, +0.3] | -0.7 [-1.6, +0.2] | -0.3 [-1.2, +0.5] | +4.2* [+3.5, +5.0] |
| mlp_clean+diff | -1.7* [-2.3, -1.0] | -0.2 [-0.9, +0.4] | -1.5* [-2.2, -0.9] | -1.6* [-2.3, -0.9] | -0.6 [-1.4, +0.3] | -1.3* [-2.0, -0.6] | -0.6 [-1.2, +0.1] | +2.3* [+1.7, +3.0] |

## Aligned-trained − clean-trained, AUROC, 7-generator mean, paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Detector | Clean | JPEG Q50 | Resize 128 | Resize 64 | Resize 32 | 128 + JPEG 50 | Format q96 | Format 256+q96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rigid | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| logreg_clean | -0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.003] | -0.002 [-0.004, +0.000] | -0.002 [-0.005, +0.000] | -0.002 [-0.004, +0.001] | +0.001 [-0.001, +0.004] | +0.002* [+0.000, +0.004] | +0.007* [+0.005, +0.009] |
| logreg_clean+diff | -0.004 [-0.007, +0.000] | -0.003 [-0.009, +0.003] | -0.005* [-0.010, -0.001] | -0.012* [-0.019, -0.006] | -0.002 [-0.010, +0.006] | -0.007* [-0.014, -0.000] | +0.001 [-0.003, +0.005] | +0.020* [+0.016, +0.023] |
| mlp_clean+diff | -0.006* [-0.008, -0.004] | -0.014* [-0.018, -0.010] | -0.005* [-0.008, -0.003] | -0.008* [-0.012, -0.004] | -0.005 [-0.010, +0.001] | -0.019* [-0.023, -0.014] | -0.003* [-0.005, -0.001] | +0.011* [+0.009, +0.013] |

## Class recall of aligned-trained probes (7-generator mean): real / generated

| Detector | Clean real / gen. | JPEG Q50 real / gen. | Format 256+q96 real / gen. |
|---|---:|---:|---:|
| rigid | 66% / 80% | 48% / 81% | 79% / 66% |
| logreg_clean | 76% / 84% | 69% / 80% | 81% / 79% |
| logreg_clean+diff | 76% / 87% | 88% / 57% | 87% / 80% |
| mlp_clean+diff | 80% / 87% | 91% / 58% | 89% / 81% |

## mlp_clean+diff balanced accuracy per held-out generator

| Held-out generator | Clean (clean-tr.) | Clean (aligned-tr.) | 256+q96 (clean-tr.) | 256+q96 (aligned-tr.) |
|---|---:|---:|---:|---:|
| BigGAN | 92.8% | 89.9% | 94.5% | 93.5% |
| VQDM | 86.1% | 83.7% | 87.2% | 87.8% |
| SD v1.5 | 85.6% | 84.0% | 81.0% | 83.6% |
| Wukong | 86.5% | 86.2% | 80.9% | 85.2% |
| ADM | 76.6% | 75.6% | 74.2% | 77.7% |
| GLIDE | 91.5% | 85.4% | 91.1% | 90.5% |
| Midjourney | 78.1% | 80.6% | 68.6% | 75.6% |
