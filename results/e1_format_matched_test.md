# E1: format-matched test set

Probes from the main leave-one-generator-out runs (fit on clean seen-generator embeddings, threshold from the clean seen-generator validation split), tested on the held-out generator with **both** classes put through the same format pipeline. **Format q96** = re-saved as JPEG quality 96 4:4:4 (the real photos' own setting) at native size; **Format 256+q96** = resized to shorter side 256, then JPEG q96, so compression and size no longer differ between classes. Real photos are thereby JPEG-compressed twice, generated images once. JPEG Q50 (both classes, from the main runs) is shown for reference. Mean over the 7 held-out generators, worst generator in brackets.

## Balanced accuracy

| Detector | Clean | Format q96 | Format 256+q96 | JPEG Q50 |
|---|---:|---:|---:|---:|
| rigid | 75.9% [64.9] | 73.5% [61.6] | 72.5% [55.3] | 69.1% [57.2] |
| logreg_clean | 80.5% [68.6] | 78.5% [67.5] | 78.9% [63.7] | 74.8% [62.4] |
| logreg_clean+diff | 83.0% [74.4] | 80.2% [69.4] | 79.1% [64.0] | 71.5% [59.7] |
| mlp_clean+diff | 85.3% [76.6] | 83.3% [71.6] | 82.5% [68.6] | 74.7% [64.2] |

## AUROC

| Detector | Clean | Format q96 | Format 256+q96 | JPEG Q50 |
|---|---:|---:|---:|---:|
| rigid | 0.839 [0.717] | 0.816 [0.694] | 0.825 [0.666] | 0.770 [0.606] |
| logreg_clean | 0.883 [0.768] | 0.867 [0.752] | 0.875 [0.734] | 0.827 [0.694] |
| logreg_clean+diff | 0.905 [0.817] | 0.882 [0.771] | 0.889 [0.770] | 0.804 [0.694] |
| mlp_clean+diff | 0.927 [0.844] | 0.911 [0.807] | 0.916 [0.823] | 0.854 [0.756] |

## Change in balanced accuracy (pp), 7-generator mean, paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Detector | Format q96 − clean | Format 256+q96 − clean |
|---|---:|---:|
| rigid | -2.4* [-3.8, -1.1] | -3.5* [-4.7, -2.2] |
| logreg_clean | -2.1* [-3.3, -0.8] | -1.6* [-2.8, -0.3] |
| logreg_clean+diff | -2.8* [-4.0, -1.6] | -3.9* [-5.0, -2.7] |
| mlp_clean+diff | -2.0* [-3.1, -0.8] | -2.8* [-4.0, -1.7] |

## Change in AUROC, 7-generator mean, paired bootstrap (2000 replicates per holdout); * = 95% interval excludes 0

| Detector | Format q96 − clean | Format 256+q96 − clean |
|---|---:|---:|
| rigid | -0.024* [-0.037, -0.011] | -0.014* [-0.027, -0.001] |
| logreg_clean | -0.016* [-0.027, -0.005] | -0.008 [-0.019, +0.003] |
| logreg_clean+diff | -0.023* [-0.034, -0.013] | -0.016* [-0.026, -0.007] |
| mlp_clean+diff | -0.016* [-0.025, -0.007] | -0.011* [-0.020, -0.002] |

## Class recall (7-generator mean): real / generated

| Detector | Clean real / gen. | Format q96 real / gen. | Format 256+q96 real / gen. |
|---|---:|---:|---:|
| rigid | 81% / 71% | 81% / 66% | 89% / 55% |
| logreg_clean | 83% / 78% | 83% / 73% | 87% / 71% |
| logreg_clean+diff | 87% / 79% | 87% / 73% | 93% / 65% |
| mlp_clean+diff | 87% / 83% | 88% / 79% | 93% / 72% |

## mlp_clean+diff AUROC per held-out generator

| Held-out generator | Clean | Format q96 | Format 256+q96 |
|---|---:|---:|---:|
| BigGAN | 0.983 | 0.971 | 0.986 |
| VQDM | 0.935 | 0.918 | 0.951 |
| SD v1.5 | 0.927 | 0.918 | 0.898 |
| Wukong | 0.942 | 0.937 | 0.922 |
| ADM | 0.844 | 0.807 | 0.857 |
| GLIDE | 0.982 | 0.958 | 0.975 |
| Midjourney | 0.879 | 0.869 | 0.823 |
