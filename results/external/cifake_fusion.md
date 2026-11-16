# fusion_detector_dinov2_l20_dinov3_l24 on cifake_fusion

Shipped threshold: fused_score >= 0.6878. Skipped files: 0.

| Set | AUROC | BA @ shipped | BA @ test GMM | BA @ oracle | Real recall | Generated recall |
|---|---:|---:|---:|---:|---:|---:|
| all (10000 real, 10000 generated) | 0.835 | 70.7% | 75.5% | 75.6% | 52.2% | 89.2% |
| FAKE vs all real | 0.835 | 70.7% | 75.5% | 75.6% | 52.2% | 89.2% |
| real: REAL |  |  |  |  | 52.2% |  |
