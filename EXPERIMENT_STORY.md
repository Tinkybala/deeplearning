# The experiment story: detecting images from unseen generators, robustly

This document links the experiments on all three branches into one argument: what each step asked, why, what it found, and how that motivated the next. Numbers are copied from the result files cited in each section. Detailed tables stay in `README.md`, `REPORT.md` and `results/`.

**Project question (direction J).** Can a detector trained on images from known generators recognise images from a generator it has never seen? Does it stay accurate after the everyday degradations of shared images, namely JPEG compression and downscaling?

**Who did what.**
- **Michael** (`michael`): RIGID, the training-free starting point; probes on its features; real-only anomaly detection; cross-dataset tests on CIFAKE, Synthbuster/RAISE, COCO, Unsplash, FFHQ/LSUN.
- **Tyrone** (`tyrone`): the fine-tuned ResNet-18 baseline, its augmentation ablation and its 7-generator sweep.
- **Seunghwan** (`seunghwan/dinov2-genimage-holdout`): frozen-DINOv2 detectors on Tyrone's protocol, followed by stress tests, robustness fixes and model selection.

---

## The argument in one page

| Step | Question | Answer | Leads to |
|---|---|---|---|
| 1. RIGID (M) | Can a frozen foundation model detect fakes with no training? | Sometimes. AUROC 0.96–0.99 on FFHQ/LSUN, but about 0.5 on COCO SD3/SD3.5/Midjourney 6 and ImageNet DiT. | Add a small trained head. |
| 2. Probe on RIGID features (M) | Does a little supervision fix it? | Yes within a dataset: 0.90–0.93 AUROC on an unseen generator. Only 0.65–0.76 on a different dataset. Real-only anomaly detection is no better than RIGID. | Train on fakes, but measure generalisation under control. |
| 3. ResNet-18 baseline (T) | How does a standard fine-tuned CNN transfer? | GLIDE held out: 85% balanced accuracy. Over all 7 holdouts: 72.2%, ranging from 55% to 90%. Augmentation fixes only the degradation it imitates. | One shared, rotating protocol for every model. |
| 4. Frozen DINOv2 on the same protocol (S) | Frozen features vs fine-tuning? | MLP on clean + noise-change features: **85.3%** vs 72.2%. Ahead by 9–22 pp in every condition. | Is 85% real? |
| 5. Stress-testing the headline (S) | Family leakage, file-format shortcut, seed noise? | Each costs a little: family-level 84.2%, format-matched 82.5%, seed SD 0.2–0.4 pp. The headline survives at about 84%. | It breaks under degradation: 61.6% at 32 px. |
| 6. Robustness (S, T's idea) | Routing, augmentation, better features? | Routing +1.5 pp; augmentation +1.7 pp; block-20 features plus routing **79.2%**. | Which choices were real and which were luck? |
| 7. Honest model selection (S) | Does seen-generator validation pick the right model? | No, it rewards overfitting. Nested leave-one-generator-out selection confirms block 20 and gives +2.3 pp clean. | Is what remains a feature problem or a threshold problem? |
| 8. Thresholds (S) | How much is threshold drift? | For block 20, most of it. A label-free threshold gives **80.2%** without knowing the degradation. | The final detector. |
| 9. Final detector (S) | Combine block 20 with augmented training; can single images match the batch threshold? | Augmentation works on block 20: +4.6 pp. One fixed threshold gives **81.3%** on single images; a batch threshold gives 82.9%. | Can another backbone or more pixels beat it? |
| 10. CLIP blocks and native resolution (S) | Do intermediate CLIP blocks or native-resolution tiles help? | CLIP block 16 is 92% clean but reads JPEG as "real" (59–67% format-matched); fusion adds about 1 pp. Native tiles give +4.4 pp clean and +5.0 pp format-matched, but drift under JPEG and may use a scale cue. | A size-matched test, then augmented training on native tiles. |

(M = Michael, T = Tyrone, S = Seunghwan)

---

## 1. Training-free RIGID (Michael)

**Rationale.** A detector that needs no fake images can't overfit to known generators, which makes it the natural first try for unseen generators.

RIGID embeds an image and a slightly noisy copy with frozen DINOv2 ViT-L/14. Real photos are assumed to be more stable under that noise.

**Findings** (`anomaly_detection.ipynb`, `scripts/diagnose.py`). RIGID works very well on some sources, at AUROC 0.957–0.990 on FFHQ (LDM, StyleGAN2) and 0.867 on Unsplash/SD. On others it is close to chance: 0.49–0.59 on COCO SD3, SD3.5 and Midjourney 6, 0.52 on ImageNet DiT and 0.57 on LSUN-ADM.

The diagnosis sweep (squash vs crop vs native resolution, matched size, JPEG round-trips) shows that its score depends strongly on resolution and compression, so preprocessing is part of the method.

**Takeaway.** The noise-stability signal is real but not universal. It needs either supervision or a different signal.

## 2. Probes on RIGID features, and real-only detection (Michael)

**Rationale.** Keep the frozen backbone, but learn which embedding directions matter. The probe uses the clean embedding plus its change under noise (*clean + diff*).

**Findings** (`scripts/train_probe.py`; summary in `RIGID_probe_single_image.ipynb`).

| Probe test | Plain RIGID (AUROC) | clean + diff probe (AUROC) |
|---|---:|---:|
| Same dataset, unseen images | 0.60–0.88 | 0.94–0.98 |
| Unseen generator, same dataset | 0.60–0.88 | ~0.90–0.93 |
| A different dataset | — | 0.65–0.76 |

A probe trained on ImageNet + Synthbuster reaches 0.80–0.90 AUROC on new test collections: COCO, Unsplash, and LSUN/FFHQ/DiT/GigaGAN.

The anomaly-detection notebook asked whether modelling only real images would avoid "seen-generator" bias. Its Mahalanobis and kNN scores on real-only features did no better than RIGID (0.77 / 0.56 / 0.88 against RIGID's 0.77 / 0.54 / 0.87). The supervised probes stayed best.

**Takeaway.** Some fake supervision is needed, and generalisation to a new generator is the central risk. Generalisation to a new dataset is a bigger risk still. That needs a controlled benchmark in which only the generator changes.

## 3. ResNet-18 baseline and augmentation ablation (Tyrone)

**Rationale.** Establish a standard supervised baseline and a strict protocol on GenImage. The protocol has a held-out generator, a seed-42 split, a threshold chosen on seen-generator validation, and six test conditions: clean, JPEG Q50, resize to 128/64/32 px, and 128 px + JPEG.

**Findings** (`REPORT.md`).
- **GLIDE held out:** 85.1% balanced accuracy on clean images, but 56.7% under JPEG Q50 and 50–51% at 64 and 32 px.
- **The augmentation ablation is distortion-specific:**
  - JPEG augmentation lifts JPEG Q50 from 56.7% to 76.0% at no clean cost.
  - Resize augmentation lifts 64 px by +15.4 pp, but costs 9.4 pp on clean images.
  - Combining both helps most at low resolution, though not at 128 + JPEG.
- **The 7-generator sweep** averages **72.2%** clean (AUROC 0.803), with VQDM at 55.3% and GLIDE at 90.4%.

**Takeaway.** One holdout is not enough, and robustness comes only for the degradations seen in training. Every later comparison uses this protocol on all 7 holdouts.

## 4. Frozen DINOv2 on the shared protocol (Seunghwan)

**Rationale.** Run Michael's detectors (RIGID, linear probes, clean + diff MLP) on exactly Tyrone's split, seed, conditions and threshold rule, so that everything is directly comparable.

**Findings** (`results/logo_summary.md`; ResNet re-scored from Tyrone's checkpoints in `results/resnet_dinov2_comparison.md`). Mean balanced accuracy over the 7 holdouts:

| Detector | Clean | JPEG Q50 | Resize 64 | Resize 32 |
|---|---:|---:|---:|---:|
| ResNet-18, fine-tuned (Tyrone) | 72.2% | 63.3% | 51.7% | 52.4% |
| RIGID (no training) | 75.9% | 69.1% | 57.3% | 50.0% |
| Logistic regression, clean embedding | 80.5% | 74.8% | 73.1% | 63.8% |
| **MLP, clean + diff** | **85.3%** | 74.7% | 74.1% | 61.6% |

The frozen probe beats the fine-tuned ResNet by 9–22 pp in every condition. Tyrone's published numbers reproduce to within 0.1 pp from his checkpoints.

ResNet looks good on the generators it trained on (86% validation accuracy) but transfers poorly (72%). That is the first sign of a theme that recurs: **fit to seen generators is not generalisation.**

**Takeaway.** Frozen foundation features plus a small head generalise much better than end-to-end fine-tuning. The clean + diff MLP becomes the reference detector.

## 5. Is 85% real? Stress tests (Seunghwan)

**Rationale.** Before building on the headline, rule out the obvious ways it could be inflated.

- **Family leakage (E3, A1).** SD v1.5 and Wukong are both Stable Diffusion models. Holding out the whole family costs 2.9–5.8 pp on them. GLIDE and ADM, both OpenAI pixel diffusion models, don't help each other.
  - The family-level headline is **84.2%** (−1.1 pp).
- **File-format shortcut (failure analysis, E1, E2).** In GenImage every real photo is a JPEG and every generated image a PNG at a fixed size. Detectors partly learn "compressed means real".
  - With both classes put through the same format, the MLP drops to **82.5%** (−2.8 pp). Midjourney drops the most, from 78.1% to 68.6%.
  - Training on format-matched images recovers most of this, to 84.8%.
  - So about 2–3 pp of the headline comes from the format, and the format-independent figure is about 84–85%.
  - ResNet is no more format-dependent (−0.7 pp).
- **Seed noise (A4).** Over 5 training seeds, the standard deviation of the 7-generator mean is 0.2–0.4 pp. Every key comparison keeps its sign on all seeds.

**Takeaway.** The headline survives at about 84%. The real weakness is degradation: 74.7% under JPEG Q50, 74.1% at 64 px, 61.6% at 32 px.

## 6. Robustness: routing, augmentation, better features (Seunghwan; augmentation idea from Tyrone)

- **Routing (E4, A3).** The clean + diff MLP is best on clean images and the clean-only probe holds up best under heavy degradation, so a router picks per degradation.
  - Its choice is judged on the seen generators' validation images, degraded the same way: **75.7%** mean over the six conditions.
  - With the degradation type predicted from the image (83% accurate): 75.4%.
- **Augmentation, at scale** (`results/augmented_probes.md`). Tyrone's JPEG/resize augmentations were applied to DINOv2 training images for all 7 holdouts.
  - The effects are the same as for ResNet but smaller: JPEG +4.5 pp, 32 px +4.0 pp.
  - Training on clean + augmented images gives **+1.7 pp** mean for −1.1 pp clean. That single detector matches the router.
- **Backbone and layer** (`results/backbone_comparison.md`). CLIP ViT-L/14 (the UnivFD backbone) and intermediate DINOv2 blocks were both tried.
  - **Block 20** beats the final layer: 88.6% clean (+3.3 pp), +4.3 to +4.8 pp under resizing.
  - Blocks 12–16 reach 95–97% on seen-generator validation but collapse to 50–56% under JPEG.
  - CLIP is stronger on clean images (87.7%) but fragile (−10 to −17 pp under degradation).
  - The DINOv2 + CLIP average gives the best clean result, **90.0%**.
  - A router over all 22 probes averages **79.2%** over the six conditions (+5.0 pp).
- **Adding the ResNet** (`results/resnet_dinov2_comparison.md`) makes every ensemble worse (−2 to −16.5 pp). The router never picks it.

**Takeaway.** On the final layer, feature choice matters far more than augmentation; section 9 shows the two combine on block 20. However, several "best" choices (block 20, the fusion) were spotted on the test generators, so they have to be checked honestly.

## 7. Honest model selection (Seunghwan)

**Rationale.** Three results showed that seen-generator validation misleads:
- blocks 12–16 at 95–97%;
- the ResNet ensemble ranked first;
- the router's clean-ADM failure (67%).

Nested leave-one-generator-out selection chooses by accuracy on a seen generator held out from a refit. That mirrors the deployment problem.

**Findings** (`results/nested_selection.md`).
- The nested scores rank candidates almost exactly as the real test does: fusion 88.9% inner vs 90.0% outer, block 20 87.2% vs 88.6%.
- Nested clean selection gives **+2.3 pp** (picking the fusion in 4 of 7 folds).
- For robustness it picks block 20 or the block-24 patch probe: +0.9 pp as a single detector. Block 20 ranks first when thresholds are adapted.
- Selecting on clean seen-generator validation picks fragile probes: −6.9 pp mean.
- The degraded-validation router (79.2%) and the nested router (78.8%) agree, so that rule was already sound. The nested router also fixes clean ADM (67% → 77%).

**Takeaway.** Block 20 is a legitimate choice, not luck. Accuracy on seen generators is not a proxy for generalisation; that is the clearest methodological lesson of the project.

## 8. Threshold drift vs lost information (Seunghwan)

**Rationale.** Under degradation AUROC falls much less than balanced accuracy. So how much of the loss is only a threshold tuned on clean images?

**Findings** (`results/threshold_adaptation.md`).
- **For the reference MLP, little.** Even the oracle threshold recovers only 2.2 pp; the degradation destroyed information.
- **For block 20 and the CLIP fusion, a lot.** The oracle recovers 10–15 pp under JPEG Q50.
- **A label-free threshold recovers almost all of it.** It fits a two-Gaussian mixture to an unlabelled test batch's scores and lands within 0.6–0.8 pp of the oracle.
  - Block 20 with it reaches **80.2%** over the six conditions, beating the 22-probe router without knowing the degradation.
  - It is robust to 20% or 80% real batches (78.7% / 79.7%).
- **ResNet's collapse at 64 and 32 px is not a threshold problem:** its oracle reaches only 53–55%.

## 9. Final detector (Seunghwan)

**Rationale.** Combine the two strongest findings: block-20 features and augmented training. Then test whether single-image thresholds, read from the file (size and JPEG quality), can replace the batch-based label-free threshold.

**Findings** (`results/final_detector.md`). Mean balanced accuracy over the 7 holdouts:

| Detector | Clean | JPEG Q50 | Resize 64 | Resize 32 | 128 + JPEG | Mean (6 std.) |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned ResNet-18 (Tyrone) | 72.2% | 63.3% | 51.7% | 52.4% | 59.0% | 60.8% |
| Main runs' DINOv2 MLP | 85.3% | 74.7% | 74.1% | 61.6% | 68.1% | 74.2% |
| Block-20 MLP | 88.6% | 74.7% | 78.5% | 66.4% | 66.3% | 76.7% |
| **Block-20 MLP, clean + augmented training** | 88.7% | 83.2% | 80.7% | 72.3% | 76.4% | **81.3%** |
| ... with the label-free batch threshold | 90.5% | 86.2% | 81.6% | 72.8% | 78.0% | 82.9% |

- **Augmentation pays off on block 20.** It gains +4.6 pp on average, with +8.5 pp at JPEG Q50 and +10.0 pp at 128 + JPEG, at no clean cost. The same augmentation gave only +1.7 pp on the final layer. Block 20 still holds the fine detail the probe can learn to read under degradation.
- **Augmentation also removes most of the threshold drift.** A single threshold is within 2.1 pp of the oracle. File-metadata thresholds add only 0.3 pp, and routers add nothing.
- **Nested selection over all 32 augmented and clean layer probes agrees** (`results/augmented_layer_selection.md`). It picks the augmented block-12–24 concatenation or the augmented block-20 MLP, at 81.2% vs 81.3% (not significant). Augmentation also removes the early blocks' fragility: the concatenation goes from 70.2% to 82.2%.
- **Final detector:** the augmented block-20 MLP with one fixed threshold. It works on single images, needs neither the degradation type nor metadata, and averages 81.3%. That is +7.1 pp over the main runs' detector and +20.5 pp over the fine-tuned ResNet. With batches of images that share one degradation, the label-free threshold adds 1.6 pp.

## 10. Can a second backbone or native resolution beat it? (Seunghwan)

**Rationale.** Two open questions remained:
- Intermediate DINOv2 blocks beat its final layer. Is the same true of CLIP, the backbone of UnivFD?
- Every model so far sees images shrunk to 224 px. Does keeping the native pixels help?

**CLIP's intermediate blocks** (`results/clip_layer_experiments.md`, 24 candidates, nested selection):
- **The best clean accuracy so far, but largely a format shortcut.** CLIP block 16 reaches 92.1% clean. When both classes are re-saved as JPEG q96 at 256 px, it falls to 59–67%, because it reads JPEG compression as "real": generated recall goes from 86% to 18–34%. The final detector loses 3.2 pp on the same test.
- **Fusion with the final detector adds about 1 pp.** Nested selection picks the fusion with CLIP block 16 in 5 of 7 folds: +0.5 pp mean (significant), +2.5 pp clean, but −3.0 pp format-matched. The fusion with CLIP block 24 is the honest one: +1.3 pp mean with no format loss. The nested score cannot tell the two apart, because none of the six standard conditions removes the PNG/JPEG difference.
- **Lesson.** A detector chosen on clean or on the standard degradations can still be choosing a shortcut. The format-matched test from section 5 is a necessary check for every new backbone.

**Native resolution** (`results/native_tile_experiments.md`; DINOv2 block 20, clean training):
- **Averaging DINOv2 over up to four native-resolution 224 px tiles beats the downscaled crop.** It gains +4.4 pp clean (88.6% → 93.0%) and +5.0 pp format-matched, so this gain survives the format check that CLIP failed.
  - ADM gains the most (+9.6 pp clean).
  - Midjourney, at 1024 px, is the only loss (−3.2 pp).
- **Under JPEG Q50 the gain turns into threshold drift.** Balanced accuracy is −2.6 pp, with AUROC unchanged. That is the pattern of section 8, which augmented training fixed for the downscaled crop.
- **Open caveat: a possible scale cue.** The native views keep each image's size, which differs by class in GenImage. Real photos (about 375 × 500) are seen zoomed in, while 256 px fakes look almost the same as in the downscaled crop. The probe could learn "native photo scale = real"; this fits the largest gains being on the 256 px generators ADM and VQDM. A size-matched test is needed before this gain is trusted.

**Takeaway.** CLIP adds little that is not a shortcut. Native resolution is the most promising direction left, but it is unproven until a size-matched test rules out the scale cue. Only then is it worth training the final detector's recipe (block 20, augmented training) on native tiles.

---

## What the project shows

1. **Frozen foundation features plus a small head generalise to unseen generators far better than a fine-tuned CNN:** 85% vs 72% on clean images, and better in every degradation.
2. **Leave-one-generator-out overstates generalisation a little.** Family leakage and the PNG/JPEG shortcut cost about 1–3 pp each; the honest figure is about 84%.
3. **Robustness needs the right features first; then augmentation and thresholds pay off.**
   - An intermediate DINOv2 block keeps the signal under JPEG and resizing.
   - On it, augmented training adds 4.6 pp, against 1.7 pp on the final layer, and removes most of the threshold drift.
   - A label-free threshold absorbs what remains.
4. **Choose models by performance on a held-out generator, never by fit to the training generators.** Validation on seen generators repeatedly picked the wrong model: early layers, the ResNet ensemble, the clean-ADM router choice.

## Limits and next steps

- **The cross-dataset gap is the largest open risk.** Michael's probes fall to 0.65–0.76 AUROC on a different dataset. All GenImage results share ImageNet real photos and 2021–2023 generators. A test on newer generators (FLUX, SDXL, SD3) and other real-photo sources is the natural next experiment.
- **Statistical scope.** One data split and one real-photo source. The real test photos are shared across holdouts, so the 7-generator means are not independent samples. Bootstrap intervals cover image sampling only.
- **Label-free thresholds need a batch of images that share one degradation.** The final single-image detector doesn't need them (section 9). In GenImage, file metadata partly reveals the class: generated images are never JPEG, and BigGAN's are all 128 px. That limits how far metadata rules can be trusted on this benchmark.
- **Cross-dataset test of the final detector.** `models/final_detector_dinov2_l20_mlp.pt` (trained on all 7 GenImage generators) and `scripts/score_final_detector.py` are ready to run on Michael's COCO, Unsplash, FFHQ/LSUN and Synthbuster sets.
- **Further models.**
  - A size-matched test of native-resolution tiles (section 10), then augmented training on them.
  - DINOv3 needs its gated weights; SigLIP 2 has not been tried.
  - Any new backbone must also pass the format-matched test, which CLIP's intermediate blocks failed.
