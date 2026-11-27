# Benchmarks: the published bar to beat

Part of the [dataset survey](datasets.md). Each project's `baselines.yaml` takes its numbers from here, under the protocol stated for each benchmark.

Researched 2026-11-26. I read every number below on the source linked next to it: the paper itself, its arXiv or PMC copy, or the official leaderboard page. Where I could not reach a primary source, the entry says **not verified** or **not found** and gives no number. A number marked "(computed)" is arithmetic I did on values printed in the linked source.

Three warnings apply to every section:

- **A leaderboard score and a paper score are different things.** Kaggle private-leaderboard scores come from hidden labels scored once. Paper scores usually come from a local split, often with model selection that touched the test set.
- **"SOTA" claims rarely share a protocol.** Before you compare, match the split, the label set, the image size, the use of external or pre-training data, and whether the score is a single model or an ensemble.
- **Papers with Code no longer exists.** Its archived leaderboards mix protocols, so I used none of them as a source.

---

### 1. MedMNIST v2: PneumoniaMNIST (also ChestMNIST and BreastMNIST)

**Metric and split.** Test-set AUC and ACC on the official `train/val/test` arrays in the `.npz` files.

- **PneumoniaMNIST:** binary (pneumonia vs normal), 4,708 / 524 / 624 images. The source is Kermany's pediatric CXR set. The authors split the source training set 9:1 into train and val and use the source test set as test. Images are center-cropped and resized to 28×28. ([MedMNIST v2, arXiv 2110.14795](https://arxiv.org/abs/2110.14795))
- **ChestMNIST:** multi-label over 14 findings, 78,468 / 11,219 / 22,433 images. It uses "the official data split" of NIH ChestX-ray14. ([arXiv 2110.14795](https://arxiv.org/abs/2110.14795))
- **BreastMNIST:** 780 breast ultrasound images. Normal and benign are merged against malignant to make a binary task, split 7:1:2 into 546 / 78 / 156. ([arXiv 2110.14795](https://arxiv.org/abs/2110.14795))

**MedMNIST v2 paper, Table 3** (Yang et al., Sci Data 2023). I read these values on the arXiv copy and on [medmnist.com](https://medmnist.com/), and the two match. nature.com redirected to a login page and I could not read it.

| Method | Pneumonia AUC / ACC | Chest AUC / ACC | Breast AUC / ACC |
|---|---|---|---|
| ResNet-18 (28) | 0.944 / 0.854 | 0.768 / 0.947 | 0.901 / 0.863 |
| ResNet-18 (224) | 0.956 / 0.864 | 0.773 / 0.947 | 0.891 / 0.833 |
| ResNet-50 (28) | 0.948 / 0.854 | 0.769 / 0.947 | 0.857 / 0.812 |
| ResNet-50 (224) | 0.962 / 0.884 | 0.773 / 0.948 | 0.866 / 0.842 |
| auto-sklearn | 0.942 / 0.855 | 0.649 / 0.779 | 0.836 / 0.803 |
| AutoKeras | 0.947 / 0.878 | 0.742 / 0.937 | 0.871 / 0.831 |
| Google AutoML Vision | **0.991 / 0.946** | **0.778 / 0.948** | **0.919 / 0.861** |

Source: [arXiv 2110.14795, Table 3](https://arxiv.org/abs/2110.14795) and [medmnist.com](https://medmnist.com/). In this table, "224" means the 28×28 image upsampled to 224 ("resized from 28"), not a native 224 image.

**MedMNIST+ (Doerrich et al., arXiv 2404.15786 v3).** Native 28, 64, 128 and 224 images. Each value is the mean ± std over 3 seeds, with ACC at a fixed 0.5 threshold. Every resolution is zero-padded to 224×224, not interpolated. Training runs 100 epochs with early stopping on val. ([arXiv 2404.15786](https://arxiv.org/abs/2404.15786))

- **PneumoniaMNIST, best overall:** DINO ViT-B/16 linear probe at 224 reaches AUC 98.92±0.08 and ACC 91.56±0.27. The best end-to-end model is VGG16 at 64, with AUC 98.69±0.16. ([arXiv 2404.15786, supplementary PneumoniaMNIST table](https://arxiv.org/abs/2404.15786))
- **ChestMNIST, best AUC:** DenseNet-121 end-to-end at 224 reaches 82.22±0.14. ACC sits near 94.7–94.8 for every model. ([Supp. Table S3](https://arxiv.org/abs/2404.15786))
- **BreastMNIST, best AUC:** AlexNet end-to-end at 128 reaches 93.97±1.36 with ACC 90.60±1.09. ([Supp. Table S2](https://arxiv.org/abs/2404.15786))

**Later papers with higher numbers** (single runs, 28→224 upsampled or 28 native, not peer-comparable to MedMNIST+):

- **MedViT-S (224)** (Manzari et al., arXiv 2302.09462, Table 3; the images are "first resized to a size of 224 x 224"):
    - PneumoniaMNIST: AUC 0.995 / ACC 0.961
    - ChestMNIST: MedViT-L reaches 0.805 / 0.959
    - BreastMNIST: MedViT-S reaches 0.938 / 0.897

    Source: [arXiv 2302.09462](https://arxiv.org/abs/2302.09462)
- **C-Mixer** (Zheng & Jia, arXiv 2304.10054, Table II; an arXiv preprint, 28×28, self-supervised pre-training on the training set):
    - PneumoniaMNIST: AUC 0.997 / ACC 0.966
    - ChestMNIST: 0.822 / 0.963
    - BreastMNIST: 0.939 / 0.899

    This is the highest PneumoniaMNIST number I found. The paper says fine-tuning "does not use the data augmentation scheme or early-stopping strategy", and it does not say how the checkpoint was chosen. Its ResNet baseline rows differ from the MedMNIST v2 table (for example, ResNet-18 (28) PneumoniaMNIST 0.957 / 0.843). ([arXiv 2304.10054](https://arxiv.org/abs/2304.10054))

**Bar to beat.**
- PneumoniaMNIST: AUC 0.997 / ACC 0.966 (C-Mixer, preprint). The peer-reviewed bar is 0.995 / 0.961 (MedViT-S). With multi-seed rigor, the bar is 98.92 / 91.56 (MedMNIST+).
- ChestMNIST AUC: 0.822, held by both C-Mixer and MedMNIST+ DenseNet-121 at 224.
- BreastMNIST AUC: 0.939 (C-Mixer) or 93.97 (MedMNIST+ AlexNet at 128).

**Reproducible locally?** Yes. Test labels ship in the `.npz` files.

**Pitfalls.**
- **Image size.** State "28", "28 upsampled to 224" or "native 224 (MedMNIST+)". The three give different results, and MedMNIST+ zero-pads where others interpolate.
- **ChestMNIST ACC means almost nothing.** It is per-label accuracy on a heavily negative label matrix, so every model scores about 0.947. Report AUC.
- **Variance.** BreastMNIST has 156 test images and PneumoniaMNIST has 624. A one-seed gain of 0.5 ACC points is within seed noise; MedMNIST+ stds reach 1–3 points. Report the mean ± std over at least 3 seeds and select checkpoints on val only.
- **AutoML Vision.** Google AutoML Vision leads the v2 table on PneumoniaMNIST, so compare against it as well as against the ResNets.

---

### 2. NIH ChestX-ray14

**Metric and split.** Mean AUROC over the 14 findings on the official patient-wise test list (`test_list.txt`, 25,596 images). A late-2026 preprint describes this test set as 2,797 patients ([arXiv 2611.13014](https://arxiv.org/abs/2611.13014)). The usual protocol takes `train_val_list.txt` (86,524 images, computed as 75,312 + 11,212) and carves out a val set by patient. Foundation Ark lists the split as 75,312 / 11,212 / 25,596 ([PMC11095392](https://pmc.ncbi.nlm.nih.gov/articles/PMC11095392)).

**Results on the official split:**

| Result | mean AUROC | Source |
|---|---|---|
| Guendel et al. (location-aware DenseNet-121), as cited | 0.807 | [Baltruschat et al., Sci Rep 2019](https://www.nature.com/articles/s41598-019-42294-8), read via [Europe PMC PMC6476887](https://europepmc.org/article/PMC/PMC6476887) |
| Baltruschat et al., ResNet-38-large-meta | 0.806 | same |
| SwinCheX (3-layer MLP head) | 0.810 (abstract; split not stated on the abstract page) | [arXiv 2206.04246](https://arxiv.org/abs/2206.04246) |
| Xiao et al. (WACV 2023), ViT-B/16 MAE pre-trained on 0.5M X-rays | 83.0 | [lambert-x/medical_mae README](https://github.com/lambert-x/medical_mae). The repo ships `test_official.txt` / `train_official.txt` / `val_official.txt` ([data_splits](https://github.com/lambert-x/medical_mae/tree/main/data_splits/chestxray)) |
| Foundation Ark-6 (MICCAI 2023) | 83.05±0.09 | [PMC11095392, Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11095392) |
| EVA-X-S (22M) | 83.3 | [arXiv 2405.05237](https://arxiv.org/abs/2405.05237). Split given as "75k/11k/25k", consistent with official |
| EVA-X-S reproduced independently, official split | 0.828 (best ensemble of 5 ImageNet backbones: 0.817) | [arXiv 2611.13014](https://arxiv.org/abs/2611.13014) |

**Bar to beat on the official split:** about 0.833 (EVA-X-S, self-reported). An independent rerun gives 0.828. Every result at 0.83 or above uses chest X-ray pre-training on several hundred thousand external images.

**CheXNet.** It is **not on the official split.** It used a random patient-wise 70/10/20 split ([arXiv 1711.05225](https://arxiv.org/abs/1711.05225)). Its 14 per-class AUROCs average 0.841 (computed from Table 1 of the same paper). Baltruschat et al. show that the same model loses AUC when moved to the official split ([PMC6476887](https://europepmc.org/article/PMC/PMC6476887)).

**Not verified.**
- A medRxiv preprint claims a mean ROC-AUC of 0.940 on ChestX-ray14 with a 3-model ensemble ([10.1101/2025.10.25.25338784](https://www.medrxiv.org/content/10.1101/2025.10.25.25338784v1)). medRxiv returned 403, so I could not check its split. Do not use it as a bar.
- The Ark+ paper (Nature 2025, [10.1038/s41586-025-09079-8](https://www.nature.com/articles/s41586-025-09079-8)) is likely the strongest recent foundation model, but nature.com would not load. Its ChestX-ray14 number is **not found**.

**Google's adjudicated expert labels** ([Google Cloud dataset page](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest); access through a Google form):

- **Four findings (airspace opacity, pneumothorax, nodule/mass, fracture).** 1,962 test and 2,412 val images, each adjudicated by 3 radiologists.
    - Majkowska et al. (Radiology 2020) report these AUCs on ChestX-ray14: pneumothorax 0.94, nodule/mass 0.91, airspace opacity 0.94, fracture 0.81 ([PubMed 31793848](https://pubmed.ncbi.nlm.nih.gov/31793848/)). I read them in the PubMed abstract because pubs.rsna.org returned 403.
- **All 14 findings plus normal/abnormal.** 810 PA test images, majority vote of 3 of 5 radiologists.
    - Nabulsi et al. (Sci Rep 2021) report normal-vs-abnormal AUC 0.94 (95% CI 0.93–0.96) ([PMC8410908](https://pmc.ncbi.nlm.nih.gov/articles/PMC8410908/)).
    - Mean 14-finding AUROC on these labels: **not found.**

**Reproducible locally?** Yes. The images, labels and official split files are public. The Google labels are public after you request access.

**Pitfalls.**
- **Splits.** Random or patient-wise custom splits inflate AUROC by about 3 points versus the official split; CheXNet is the standard example.
- **Label noise.** The NLP-mined labels are noisy. Scores against Google's adjudicated labels are a different benchmark.
- **Leakage through pre-training.** Pre-training corpora such as Ark, MAE and EVA-X often include ChestX-ray14 itself, so check that no official test image appears in the pre-training data.
- **Inconsistent "No Finding" and view handling.** Some papers drop AP views or "No Finding", or evaluate fewer than 14 labels.

---

### 3. CheXpert

**Metric and split.** Mean AUROC over the 5 competition findings (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion), on the hidden test set of 500 studies from 500 unseen patients.

- **Test ground truth:** 8 radiologists labeled each study; the majority vote of 5 is ground truth and the other 3 are the human benchmark.
- **Validation set:** 200 studies, with the consensus of 3 radiologists.

Source: [CheXpert competition page](https://stanfordmlgroup.github.io/competitions/chexpert/)

**Leaderboard top entries** ([CheXpert competition page](https://stanfordmlgroup.github.io/competitions/chexpert/)):

| Rank | Model | AUC |
|---|---|---|
| 1 | DeepAUC-v1 (ensemble), Aug 31 2020 | 0.930 |
| 2 | Hierarchical-Learning-V1 (ensemble), Vingroup Big Data Institute | 0.930 |
| 3 | Conditional-Training-LSR (ensemble) | 0.929 |

All top-10 entries are ensembles, between 0.927 and 0.930.

**Bar to beat:** 0.930, an ensemble.

**Reproducible locally?** Yes, now. The page says the competition "is now closed" and the test labels and image links are on GitHub ([competition page](https://stanfordmlgroup.github.io/competitions/chexpert/)). Because the test set is open, nobody else checks your numbers: you must hold yourself to never tuning on it.

**Pitfalls.**
- **Validation reported as test.** Many papers report the 234-image (200-study) validation set and call it "test".
    - Foundation Ark lists CheXpert test as 234 images ([PMC11095392, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11095392)), with Ark-6 at 89.14.
    - Xiao et al.'s 89.3 CheXpert mAUC ([medical_mae](https://github.com/lambert-x/medical_mae)) has no stated test set.
    - EVA-X still says the official test set "is not public" ([arXiv 2405.05237](https://arxiv.org/abs/2405.05237)).
    - None of these numbers is comparable to 0.930.
- **Frontal-only images and other label sets.** CheXFound reports AUROC 0.908 on 518 test images (the frontal images), with its own 202-image val set ([arXiv 2502.05142](https://arxiv.org/abs/2502.05142)). It may not use the 5-finding average; that is not verified.
- **Uncertain-label policy.** U-Ones, U-Zeros or label smoothing change results; state which you use.
- **Model selection on val.** With only 200 studies, choosing checkpoints on val overfits.

---

### 4. RSNA Pneumonia Detection Challenge (2018)

**Metric** ([Kaggle evaluation page](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview/evaluation)):

- **Per-image score:** for each image, at IoU thresholds 0.40 to 0.75 in steps of 0.05, compute TP/(TP+FP+FN), then average over the thresholds.
- **Final score:** the mean over all test images.
- **Negative images:** if an image has no ground-truth box, any prediction scores that image **0**.

This is not COCO or VOC mAP. Kaggle calls it "mean average precision", but it is a per-image average with false positives penalized on negatives.

**Split.** A two-stage competition. The stage 2 training set is the stage 1 train plus the stage 1 test, and the stage 2 test is "new, unseen images" ([data page](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)). The private leaderboard uses about 99% of the test data.

**Private leaderboard** ([Kaggle leaderboard](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/leaderboard)):

| Place | Team | Score |
|---|---|---|
| 1 | Ian Pan & Alexandre Cadrin | **0.25475** |
| 2 | Dmytro Poplavskiy [ods.ai] | 0.24781 |
| 3 | Phillip Cheng | 0.23908 |

The approaches of two top-placing teams are described in Pan et al., AJR 2019 ([PubMed 31120793](https://pubmed.ncbi.nlm.nih.gov/31120793/)). The abstract gives no scores.

**Paper on a standard split:** **not found.** Papers I saw used their own splits of the stage 2 training labels and COCO-style mAP@0.5. The dataset paper, Shih et al. (Radiology: AI 2019, [PubMed 33937785](https://pubmed.ncbi.nlm.nih.gov/33937785/)), gives no baseline in its abstract, and I could not read its full text (PMC served a CAPTCHA, which I did not bypass).

**Reproducible locally?**
- **Kaggle late submissions:** the competition page shows a "Late Submission" button, so you can still get official private and public scores.
- **Stage 2 test boxes:** RSNA's page offers "the adjudicated images and annotations used in the challenge" plus a mapping to NIH IDs ([RSNA page](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)). Whether that download includes the stage 2 test boxes is **not verified**.

**Pitfalls.**
- **Overlap with ChestX-ray14.** The images are a subset of NIH ChestX-ray14, so a backbone pre-trained on ChestX-ray14, or on a corpus that contains it, has seen the test images.
- **Metric implementation.** Reimplementations of the metric differ, especially the zero score for false positives on negatives. Use the Kaggle definition, or submit late.
- **Classification on the same data.** Results on "RSNA Pneumonia" image classification (AUC) are a different task on the same images.

---

### 5. VinDr-CXR / VinBigData Chest X-ray Abnormalities Detection

**Metric** ([Kaggle evaluation](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/overview/evaluation)). "PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.4", over 14 finding classes. An image with no findings must be predicted as class 14, "No finding" (`14 1 0 0 1 1`).

**Data** ([Kaggle data page](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data)):
- The competition uses 18,000 PA scans.
- The [VinDr-CXR paper (Sci Data 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9300612) uses 15,000 train scans, each labeled by 3 radiologists, and 3,000 test scans labeled by the consensus of 5. The paper has 22 local and 6 global labels and calls the Kaggle version "a slightly modified version of this dataset".
- The Kaggle private leaderboard uses about 90% of the test data.

**Private leaderboard** ([Kaggle leaderboard](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/leaderboard)):

| Place | Team | mAP@0.4 |
|---|---|---|
| 1 | ℳS²Ƒ | **0.314** |
| 2 | SZI | 0.307 |
| 3 | scumed | 0.305 |

**The VinDr-CXR paper's baseline:** **not found.** The Sci Data article's Technical Validation section covers only de-identification and labeling QC. It has no detection model, mAP or AUC ([PMC9300612](https://pmc.ncbi.nlm.nih.gov/articles/PMC9300612); the arXiv version [2012.15029](https://arxiv.org/abs/2012.15029) is the same).

**Reproducible locally?**
- The paper says it released "all images together with the labels of the training set and the test set", with test labels given as 5-radiologist consensus ([PMC9300612](https://pmc.ncbi.nlm.nih.gov/articles/PMC9300612)). The data is on [PhysioNet](https://www.physionet.org/content/vindr-cxr/) under credentialed access.
- Kaggle also shows "Late Submission".

**Pitfalls.**
- **Training ground truth comes from 3 readers per image.** How you fuse the boxes (WBF, per-radiologist, or majority) changes the scores a lot.
- **Two test sets.** The Kaggle test set and the PhysioNet test labels may not be identical ("slightly modified"; not verified), so quote either a Kaggle late-submission score or a PhysioNet-test score and say which.
- **Kaggle-specific gains.** The Kaggle score rewards a "No finding" class and a 2-class image-level filter. Papers using 22 classes, COCO mAP or a different IoU are not comparable.

---

### 6. ISIC 2020 (SIIM-ISIC Melanoma Classification)

**Metric.** "Area under the ROC curve between the predicted probability and the observed target", per image ([Kaggle evaluation](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview/evaluation)). The private leaderboard uses about 70% of the test data.

**Private leaderboard** ([Kaggle leaderboard](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/leaderboard)):

| Place | Team | AUC |
|---|---|---|
| 1 | All Data Are Ext | **0.9490** |
| 2 | aloe | 0.9485 |
| 3 | Deloitte Analytics Spain | 0.9484 |

The winning write-up (Ha, Liu & Liu, [arXiv 2010.05351](https://arxiv.org/abs/2010.05351)) reports 0.9600 cross-validation AUC and 0.9490 on the private leaderboard. It is an ensemble of EfficientNets at several input sizes, mostly image-only with some metadata models. It trains on the 2018 and 2019 ISIC data as well as 2020, and states that public leaderboard feedback was not used.

**ISIC live leaderboard (2020 task)** ([challenge.isic-archive.com/leaderboards/live](https://challenge.isic-archive.com/leaderboards/live/)). It lists 200 submissions from 36 teams. The top entry is 0.949 AUC ("temp3", marked no external data), and the next is 0.942. The ISIC home page says ISIC hosts "live challenges" for benchmarking "on an ongoing basis" ([ISIC Challenge](https://challenge.isic-archive.com/)). It now advertises the MILK10k benchmark as open for submissions. I did not log in, so whether the 2020 live task's submission form is open today is **not verified**. Kaggle still shows "Late Submission" for the competition.

**Reproducible locally?** **No.** The ISIC data page offers the 2020 test images and test metadata but no test ground truth ([ISIC data page](https://challenge.isic-archive.com/data/)). Score through a Kaggle late submission or the ISIC live leaderboard.

**Pitfalls.**
- **Grouping.** Split by `patient_id`; many patients have several lesions.
- **Duplicates.** ISIC publishes a training duplicate-image list ([data page](https://challenge.isic-archive.com/data/)); remove duplicates before you split.
- **External data.** Most top entries use ISIC 2017, 2018 and 2019 images, so say whether you do.
- **Public vs private.** The public leaderboard (30%) is noisy; the winners say it correlated poorly with CV.
- **Own splits.** Papers that report AUC on their own 2020-train split cannot be compared with 0.949.

---

### 7. CBIS-DDSM

**Official split** ([TCIA CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)):
- The split is by BI-RADS category, with 20% of cases for testing. Mass and calcification cases are split separately.
- 254 images in which the mass was not clearly seen were removed.
- Subject counts: Mass-Training 691, Mass-Test 201, Calc-Training 602, Calc-Test 151.

There is no official metric or leaderboard.

**Best published results:**

| Task and protocol | Best number | Source |
|---|---|---|
| Two-view (CC+MLO) benign/malignant, **official split** | AUC 0.8643±0.0241 (EfficientNet-B3), 0.8658 with TTA. Images resized to 1152×896. The authors call the gain over 0.8483 weakly significant (p = 0.1366) | [arXiv 2503.19945](https://arxiv.org/abs/2503.19945) |
| Same task, official split, earlier | AUC 0.8483 (5-fold CV: 0.9344) | [Petrini et al., IEEE Access 2022, arXiv 2110.01606](https://arxiv.org/abs/2110.01606) |
| Single-view, official split | AUC 0.8325±0.0171 (patch-based EfficientNet-B3) | [arXiv 2503.19945](https://arxiv.org/abs/2503.19945) |
| Whole-image classification, **own random 85:15 patient split** (376 test images) | per-image AUC 0.88 single model, 0.91 for a 4-model ensemble | [Shen et al., Sci Rep 2019, PMC6715802](https://pmc.ncbi.nlm.nih.gov/articles/PMC6715802/) |
| Mass detection (recall at fixed false positives per image) on the official Mass-Test split | **not found.** The detection papers I checked use DDSM rather than CBIS-DDSM, custom test sets, or an IoU > 0.1 mask criterion (for example [MAMM-Net, arXiv 2404.16718](https://arxiv.org/abs/2404.16718)) | none |

**Bar to beat:** AUC about 0.866 for two-view classification on the official split.

**Reproducible locally?** Yes. TCIA is public under CC BY 3.0 and the official test labels are included.

**Pitfalls.**
- **Splits.** Random splits score much higher than the official split: 0.934 under 5-fold CV versus 0.848 on the official split, for the same method ([arXiv 2110.01606](https://arxiv.org/abs/2110.01606)).
- **ROI crops vs whole images.** Classifying ROI crops is a different and much easier task than classifying whole images.
- **Task definitions vary.** Mass and calcification are sometimes merged, and the unit may be per image, per breast or per abnormality.
- **Patient IDs.** One participant can have several patient IDs (TCIA counts 1,566 real participants against 6,671 IDs), so a naive patient-level split can leak.
- **Detection metrics.** Detection papers mix FROC operating points and hit criteria.

---

### 8. LUNA16

**Metric and protocol** ([Evaluation](https://luna16.grand-challenge.org/Evaluation/), [Data](https://luna16.grand-challenge.org/Data/)):
- **CPM:** the average FROC sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives per scan. A hit is a candidate within radius = diameter/2 of the nodule center; findings on "irrelevant" annotations are ignored.
- **Data:** 888 CT scans from LIDC-IDRI with slice thickness ≤ 2.5 mm, and 1,186 reference nodules (≥ 3 mm, accepted by at least 3 of 4 radiologists).
- **Protocol:** 10-fold cross-validation over the 10 provided subsets ("The dataset is used for both training and testing dataset").
- **Tracks:** nodule detection (NDET) and false-positive reduction (FPRED, using `candidates_V2.csv`).

**Leaderboard** ([LUNA16 Results](https://luna16.grand-challenge.org/Results/)):

| Track | Place | Team | Date | CPM |
|---|---|---|---|---|
| Nodule detection | 1 | PAtech | 2 Jan 2018 | **0.951** |
| Nodule detection | 2 | JianPeiCAD | | 0.950 |
| Nodule detection | 3 | LUNA16FONOVACAD | | 0.947 |
| False-positive reduction | 1 | PATech | | **0.968** |
| False-positive reduction | 2 | LUNA16FONOVACAD | | 0.966 |

**Papers under the 10-fold protocol** ([nnDetection, MICCAI 2021, arXiv 2106.00817, Table 2](https://arxiv.org/abs/2106.00817)):
- Liu et al. (2019) with false-positive reduction: CPM 0.952.
- nnDetection without false-positive reduction: CPM 0.930, with sensitivity 0.812 at 1/8 and 0.985 at 8 false positives per scan.
- Liu et al. (2019) without false-positive reduction: 0.919.

**Bar to beat:** NDET 0.951 (leaderboard) or 0.952 (paper, with false-positive reduction); FPRED 0.968.

**Reproducible locally?** Yes. All 888 scans, the annotations and the official evaluation script are public. There is no hidden test set, so the leaderboard is self-reported 10-fold CV results that participants upload. "Teams can upload as often as they want."

**Pitfalls.**
- **Folds.** Use exactly the 10 provided subsets. Custom folds or a single held-out subset are not comparable.
- **Tracks.** FPRED scores (on given candidates) are not NDET scores.
- **Evaluation script.** Use the official script with `annotations_excluded.csv`; dropping the irrelevant-findings rule changes the false-positive counts.
- **Pre-training on LIDC-IDRI.** Pre-training on LIDC-IDRI, which is a superset of the LUNA16 scans, leaks across folds.
- **Old candidate list.** Leaderboard entries marked * used the older, smaller candidate list.

---

### 9. Kermany OCT (OCT2017, CNV / DME / drusen / normal)

**Metric and split.** 4-class accuracy on the official held-out test folder, described as 1,000 images (250 per class) against 83,484 training images ([OpticNet, arXiv 1910.05672](https://arxiv.org/abs/1910.05672)). The [Mendeley dataset (v3)](https://data.mendeley.com/datasets/rscbjbr9sj/3) says train and test are "independent patients". It does not list per-class test counts on its landing page.

**Best numbers on that test set:**
- **OpticNet-71:** test accuracy 99.80%, sensitivity 99.80, specificity 99.93 ([arXiv 1910.05672, Table IV](https://arxiv.org/abs/1910.05672)). The same table lists Xception at 99.70 and MobileNet-v2 at 99.40.
- **ReLaX:** accuracy 0.9979 on **968** test images ([arXiv 2109.02436](https://arxiv.org/abs/2109.02436)). That is a different test-set version, so it is not the same benchmark.
- **The original Kermany et al. result (Cell 2018):** **not verified.**
    - cell.com and sciencedirect.com both returned 403.
    - OpticNet's table cites InceptionV3 at 96.60 accuracy, 97.80 sensitivity and 97.40 specificity ([arXiv 1910.05672](https://arxiv.org/abs/1910.05672)).
    - ReLaX cites 0.961 ([arXiv 2109.02436](https://arxiv.org/abs/2109.02436)).
    - These are secondary citations, and they disagree.

**Bar to beat:** 99.8% (2 errors out of 1,000). The benchmark is saturated, and gains are within the 1–2 image noise floor.

**Reproducible locally?** Yes. The test folder and its labels are public on Mendeley.

**Pitfalls.**
- **Test-set versions.** Different dataset versions have 1,000-image and 968-image test sets.
- **Image-level splits.** Repartitioning the 84k images at image level instead of using the patient-independent test folder leaks adjacent B-scans of the same volume. Leakage inflated accuracy by 5% to 30% in [arXiv 2202.12267](https://arxiv.org/abs/2202.12267).
- **OCTMNIST is a different split.** OCTMNIST uses its own split and 28×28 images ([arXiv 2110.14795](https://arxiv.org/abs/2110.14795)), so do not compare it with this benchmark.
- **Mixing with other sets.** Some papers merge in the Srinivasan 2014 set or evaluate 3 classes.

---

## Numbers I could not verify on a primary source

- **Kermany et al. (Cell 2018) OCT test accuracy.** It is cited as 96.6% or 96.1%, but cell.com and ScienceDirect returned 403.
- **ChestX-ray14 0.940 mean ROC-AUC (medRxiv 2025.10.25.25338784).** medRxiv returned 403, and the split is unknown.
- **Ark+ (Nature 2025) ChestX-ray14 and CheXpert numbers.** nature.com redirected to a login page.
- **MedMNIST v2 on nature.com.** The table values come from the arXiv copy and medmnist.com instead.
- **Whether RSNA's download includes the stage 2 test boxes.** The Shih et al. full text was blocked by a CAPTCHA.
- **Whether the ISIC 2020 live leaderboard accepts new submissions today.** This needs a login.
- **Whether the Kaggle VinBigData test set matches the PhysioNet VinDr-CXR test labels.**
- **The CBIS-DDSM mass-detection state of the art on the official Mass-Test split.** Not found.
- **A paper reporting RSNA-metric mAP on a standard split.** Not found.
- **The VinDr-CXR paper's own baseline.** None exists in the published paper.
