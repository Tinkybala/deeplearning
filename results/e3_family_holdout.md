# E3: leave-one-family-out (Stable Diffusion family)

SD v1.5 and Wukong are both Stable Diffusion models. **A** = standard leave-one-generator-out (the SD sibling is in training, 6 generators). **B** = both SD generators removed (5 generators). **C** = sibling kept but one non-SD generator removed (5 generators), averaged over the 5 choices (range in brackets). B − A mixes 'no sibling' with 'one generator fewer'; B − C isolates the sibling. All settings use the same real training photos; thresholds come from each setting's own seen-generator validation split. Differences: paired stratified bootstrap (2000 replicates) on the 500 + 500 test images; * = 95% interval excludes 0.

## Balanced accuracy, clean test images

| Target | Detector | A sibling seen | B family out | C size control | B − A (pp) | B − C (pp) |
|---|---|---:|---:|---:|---:|---:|
| SD v1.5 | rigid | 64.9% | 64.9% | 64.1% [60.9%–64.9%] | +0.0 [+0.0, +0.0] | +0.8* [+0.4, +1.2] |
| SD v1.5 | logreg_clean | 79.6% | 75.5% | 79.0% [75.4%–80.4%] | -4.1* [-6.1, -2.2] | -3.5* [-5.3, -1.8] |
| SD v1.5 | logreg_clean+diff | 81.6% | 76.8% | 81.7% [80.1%–83.9%] | -4.8* [-6.4, -3.2] | -4.9* [-6.4, -3.4] |
| SD v1.5 | mlp_clean+diff | 85.6% | 82.5% | 85.2% [82.4%–87.3%] | -3.1* [-5.0, -1.3] | -2.7* [-4.1, -1.3] |
| Wukong | rigid | 68.8% | 68.8% | 67.9% [64.2%–69.0%] | +0.0 [+0.0, +0.0] | +0.9* [+0.5, +1.4] |
| Wukong | logreg_clean | 81.2% | 78.3% | 81.2% [80.1%–82.0%] | -2.9* [-5.1, -0.7] | -2.9* [-4.6, -1.3] |
| Wukong | logreg_clean+diff | 84.6% | 78.8% | 85.0% [84.2%–86.9%] | -5.8* [-7.9, -3.7] | -6.2* [-8.1, -4.4] |
| Wukong | mlp_clean+diff | 86.5% | 82.1% | 86.3% [85.6%–87.1%] | -4.4* [-6.0, -2.8] | -4.2* [-5.5, -3.0] |

## Balanced accuracy, mean over the 6 test conditions

| Target | Detector | A sibling seen | B family out | C size control | B − A (pp) | B − C (pp) |
|---|---|---:|---:|---:|---:|---:|
| SD v1.5 | rigid | 54.2% | 54.2% | 54.0% [53.1%–54.2%] | +0.0 [+0.0, +0.0] | +0.2* [+0.1, +0.4] |
| SD v1.5 | logreg_clean | 71.4% | 67.8% | 71.6% [69.6%–73.9%] | -3.5* [-4.3, -2.8] | -3.7* [-4.4, -3.0] |
| SD v1.5 | logreg_clean+diff | 68.0% | 64.2% | 68.2% [67.0%–71.5%] | -3.8* [-4.5, -3.0] | -4.0* [-4.7, -3.3] |
| SD v1.5 | mlp_clean+diff | 72.3% | 68.8% | 71.9% [70.3%–74.4%] | -3.6* [-4.4, -2.7] | -3.1* [-3.7, -2.5] |
| Wukong | rigid | 57.1% | 57.1% | 56.9% [56.1%–57.2%] | +0.0 [+0.0, +0.0] | +0.2 [-0.0, +0.4] |
| Wukong | logreg_clean | 75.0% | 72.8% | 75.4% [74.4%–76.3%] | -2.2* [-3.1, -1.3] | -2.6* [-3.2, -1.9] |
| Wukong | logreg_clean+diff | 72.4% | 68.9% | 73.1% [71.9%–75.3%] | -3.5* [-4.4, -2.6] | -4.2* [-5.0, -3.4] |
| Wukong | mlp_clean+diff | 75.2% | 71.1% | 75.1% [74.5%–76.4%] | -4.2* [-4.9, -3.4] | -4.0* [-4.6, -3.4] |

## AUROC, clean test images

| Target | Detector | A sibling seen | B family out | C size control | B − A | B − C |
|---|---|---:|---:|---:|---:|---:|
| SD v1.5 | rigid | 0.717 | 0.717 | 0.717 [0.717–0.717] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| SD v1.5 | logreg_clean | 0.874 | 0.842 | 0.873 [0.847–0.896] | -0.033* [-0.040, -0.025] | -0.031* [-0.040, -0.022] |
| SD v1.5 | logreg_clean+diff | 0.901 | 0.864 | 0.903 [0.885–0.926] | -0.037* [-0.046, -0.028] | -0.039* [-0.049, -0.029] |
| SD v1.5 | mlp_clean+diff | 0.927 | 0.901 | 0.932 [0.913–0.947] | -0.026* [-0.034, -0.019] | -0.031* [-0.038, -0.024] |
| Wukong | rigid | 0.760 | 0.760 | 0.760 [0.760–0.760] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| Wukong | logreg_clean | 0.898 | 0.869 | 0.897 [0.890–0.901] | -0.029* [-0.035, -0.022] | -0.028* [-0.036, -0.021] |
| Wukong | logreg_clean+diff | 0.918 | 0.883 | 0.921 [0.916–0.936] | -0.035* [-0.043, -0.028] | -0.039* [-0.048, -0.030] |
| Wukong | mlp_clean+diff | 0.942 | 0.913 | 0.941 [0.936–0.948] | -0.029* [-0.037, -0.021] | -0.029* [-0.035, -0.022] |

## AUROC, mean over the 6 test conditions

| Target | Detector | A sibling seen | B family out | C size control | B − A | B − C |
|---|---|---:|---:|---:|---:|---:|
| SD v1.5 | rigid | 0.543 | 0.543 | 0.543 [0.543–0.543] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| SD v1.5 | logreg_clean | 0.790 | 0.752 | 0.791 [0.768–0.814] | -0.038* [-0.041, -0.035] | -0.038* [-0.042, -0.035] |
| SD v1.5 | logreg_clean+diff | 0.742 | 0.711 | 0.747 [0.736–0.781] | -0.031* [-0.036, -0.027] | -0.036* [-0.042, -0.031] |
| SD v1.5 | mlp_clean+diff | 0.797 | 0.752 | 0.794 [0.784–0.820] | -0.045* [-0.049, -0.040] | -0.042* [-0.046, -0.039] |
| Wukong | rigid | 0.581 | 0.581 | 0.581 [0.581–0.581] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| Wukong | logreg_clean | 0.833 | 0.800 | 0.832 [0.823–0.839] | -0.032* [-0.036, -0.029] | -0.032* [-0.036, -0.028] |
| Wukong | logreg_clean+diff | 0.801 | 0.764 | 0.805 [0.788–0.829] | -0.037* [-0.041, -0.033] | -0.041* [-0.046, -0.036] |
| Wukong | mlp_clean+diff | 0.836 | 0.792 | 0.834 [0.826–0.851] | -0.043* [-0.048, -0.039] | -0.041* [-0.045, -0.038] |
