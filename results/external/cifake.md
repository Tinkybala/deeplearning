# final_detector_dinov2_l20_mlp on cifake

Shipped threshold: p_real >= 0.1714. Skipped files: 0.

| Set | AUROC | BA @ shipped | BA @ test GMM | BA @ oracle | Real recall | Generated recall |
|---|---:|---:|---:|---:|---:|---:|
| all (10000 real, 10000 generated) | 0.853 | 76.5% | 76.7% | 77.1% | 70.7% | 82.3% |
| FAKE vs all real | 0.853 | 76.5% | 76.7% | 77.1% | 70.7% | 82.3% |
| real: REAL |  |  |  |  | 70.7% |  |
