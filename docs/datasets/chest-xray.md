# Chest X-ray datasets

Part of the [dataset survey](../datasets.md).

Facts were checked on 2026-11-26 against the primary pages linked in each section. "Not
stated on <source>" means the named source does not give the fact, and the value was left
blank rather than inferred. Access classes:

- **open**: download with no account.
- **registration**: an account or form and click-through terms.
- **credentialed**: identity verification, human-subjects training, and a signed DUA.

Several Kaggle pages are rendered by JavaScript and could not be read by fetch, so the
Kaggle competition rules for RSNA and SIIM-ACR are unverified. See [Unverified](#unverified)
at the end.

---

## CheXpert

- **Name and maintainer:**
    - CheXpert v1.0, from the Stanford ML Group and Stanford AIMI (Center for Artificial Intelligence in Medicine & Imaging).
    - The images come from Stanford Hospital, inpatient and outpatient, October 2002 to July 2017.
    - Sources: https://stanfordmlgroup.github.io/competitions/chexpert/ and https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- **Tasks and labels:**
    - Multi-label classification only, with no boxes or masks.
    - 14 observations, each labeled positive (1), negative (0), uncertain (-1), or blank (not mentioned).
    - The **training labels are NLP-derived**: a rule-based labeler reads the report Impression section in three stages (mention extraction, classification, aggregation).
    - The **validation set** (200 studies) uses the consensus of 3 radiologists.
    - The **test set** (500 studies, 500 unseen patients) uses the majority vote of 5 of 8 board-certified radiologists.
    - The paper compares five ways to handle uncertain labels: U-Ignore, U-Zeroes, U-Ones, U-SelfTrained, and U-MultiClass.
    - Source: https://stanfordmlgroup.github.io/competitions/chexpert/
- **Label format:**
    - CSV (`train.csv`, `valid.csv` inside `CheXpert-v1.0-small/`), per the TFDS loader docs: https://www.tensorflow.org/datasets/catalog/chexpert
    - Test-set labels are on GitHub: https://github.com/rajpurkarlab/cheXpert-test-set-labels
- **Size:**
    - 224,316 radiographs from 65,240 patients (https://arxiv.org/abs/1901.07031).
    - The training set has 187,641 studies and the validation set 200 (datasheet, https://arxiv.org/abs/2105.03020).
    - **Patient IDs are available.** Images are grouped by patient, then by study (datasheet), with paths like `train/patient42142/study5/view1_frontal.jpg` (https://github.com/Stanford-AIMI/chexpert-plus), so patient-level splits are possible.
- **License and access:**
    - A Research Use Agreement must be read and accepted before download: "must be downloaded separately after reading and agreeing to a Research Use Agreement" (https://www.tensorflow.org/datasets/catalog/chexpert).
    - The datasheet summarizes the terms as personal, non-commercial, non-clinical research use (https://arxiv.org/abs/2105.03020).
    - Stanford AIMI says its datasets are free "for non-commercial research use", and commercial use needs a separate commercial license (https://aimi.stanford.edu/shared-datasets).
    - **The agreement's exact title is not stated on** the AIMI CheXpert page, which shows empty "License" and "Terms & Conditions" headings (https://aimi.stanford.edu/datasets/chexpert-chest-x-rays).
    - AIMI asks users to cite the dataset DOI 10.71718/y7pj-4v93.
    - The download is hosted on Azure: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
    - Access is **registration**.
- **Image format and resolution:**
    - JPEG, reduced to 256 grey levels.
    - The "large" version keeps the original DICOM size (about 440 GB).
    - The "small" version is downsampled so the long side is 320 px, about 390x320 (about 11 GB).
    - Source: datasheet, https://arxiv.org/abs/2105.03020
- **Citation:** Irvin J, Rajpurkar P, Ko M, Yu Y, Ciurea-Ilcus S, Chute C, Marklund H, Haghgoo B, Ball R, Shpanskaya K, Seekins J, Mong DA, Halabi SS, Sandberg JK, Jones R, Larson DB, Langlotz CP, Patel BN, Lungren MP, Ng AY. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI 2019. arXiv:1901.07031 (https://arxiv.org/abs/1901.07031). Official page: https://stanfordmlgroup.github.io/competitions/chexpert/
- **Pitfalls:**
    - The training labels are noisy NLP output with many uncertain (-1) labels, so the uncertainty policy changes results.
    - All images come from one institution, and the device is not recorded.
    - About 15% of patients contribute about half of the images (datasheet), so a split by image leaks patients.

## CheXpert Plus

- **Name and maintainer:**
    - CheXpert Plus, from Stanford AIMI: the CheXpert images re-released with reports, demographics, and DICOM.
    - Source: https://aimi.stanford.edu/datasets/chexpert-plus
- **Tasks and labels:**
    - Classification and report generation, with no boxes or masks.
    - 14 pathology labels **extracted automatically by CheXbert** from the reports.
    - RadGraph annotations (187,575 impression and 47,328 findings sections) and 8 demographic fields.
    - Source: https://arxiv.org/abs/2405.19538
- **Label format:**
    - CSV. The tooling references `df_chexpert_plus_240401.csv` (https://github.com/Stanford-AIMI/chexpert-plus).
    - The RadGraph file format is not stated on the sources checked.
- **Size:**
    - 223,228 unique images (paper) or 223,462 report-image pairs (AIMI page).
    - 187,711 studies and reports, and **64,725 patients**.
    - A `patient_report_date_order` field orders each patient's studies.
    - Sources: https://arxiv.org/abs/2405.19538 and https://aimi.stanford.edu/datasets/chexpert-plus
- **License and access:**
    - The Stanford University Dataset Research Use Agreement: "may not be used for any commercial purposes and is only available for research uses ... you may not distribute, publish or reproduce a copy of this dataset" (paper, https://arxiv.org/abs/2405.19538).
    - The AIMI page shows License and Terms headings with no text (https://aimi.stanford.edu/datasets/chexpert-plus).
    - The dataset DOI is 10.71718/6nvz-pm34.
    - The download is on Redivis: https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x
    - Access is **registration**.
- **Image format and resolution:**
    - DICOM (with 47 DICOM metadata elements) and PNG.
    - Resolution is not stated on the paper or the AIMI page.
- **Citation:** Chambon P, Delbrouck J-B, Sounack T, Huang S-C, Chen Z, Varma M, Truong SQH, Chuong CT, Langlotz CP. "CheXpert Plus: Augmenting a Large Chest X-ray Dataset with Text Radiology Reports, Patient Demographics and Additional Image Formats." arXiv preprint, 2024. arXiv:2405.19538 (https://arxiv.org/abs/2405.19538)
- **Pitfalls:**
    - The images overlap with CheXpert, so do not treat the two as independent cohorts.
    - The labels come from a model (CheXbert), not from radiologists.
    - Image and report counts differ slightly between the paper and the AIMI page.

## MIMIC-CXR (DICOM + reports)

- **Name and maintainer:**
    - MIMIC-CXR Database v2.1.0 (July 2024), maintained by the MIT Laboratory for Computational Physiology and hosted on PhysioNet.
    - Emergency department studies at Beth Israel Deaconess Medical Center, Boston, 2011 to 2016.
    - Sources: https://physionet.org/content/mimic-cxr/2.1.0/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC6908718/
- **Tasks and labels:**
    - Images with **free-text radiology reports**, and no boxes or masks.
    - This release has no structured labels; they are in [MIMIC-CXR-JPG](#mimic-cxr-jpg).
    - It supports report generation and, through MIMIC-CXR-JPG, classification.
- **Label format:**
    - Reports are `.txt` files.
    - CSV metadata links images, studies, patients, and (in v2.1.0) de-identified providers.
    - Source: https://physionet.org/content/mimic-cxr/2.1.0/
- **Size:**
    - 377,110 images, 227,835 studies, and **65,379 patients** (https://pmc.ncbi.nlm.nih.gov/articles/PMC6908718/).
    - **Patient IDs are available.** `subject_id` is a random value from 10,000,000 to 19,999,999, and `study_id` is from 50,000,000 to 59,999,999.
    - Dates are shifted into the years 2100 to 2200.
- **License and access:**
    - PhysioNet Credentialed Health Data License 1.5.0.
    - Access requires a credentialed PhysioNet account, CITI "Data or Specimens Only Research" training, and a signed PhysioNet Credentialed Health Data Use Agreement 1.5.0 (https://physionet.org/content/mimic-cxr/2.1.0/).
    - Access is **credentialed**.
- **Image format and resolution:**
    - DICOM.
    - Bit depth and pixel dimensions are not stated on the PhysioNet page or the Scientific Data paper.
- **Citation:** Johnson AEW, Pollard TJ, Berkowitz SJ, Greenbaum NR, Lungren MP, Deng C-y, Mark RG, Horng S. "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports." *Scientific Data* 6:317, 2019. doi:10.1038/s41597-019-0322-0. Dataset DOI (v2.1.0): 10.13026/4jqj-jw95. Official page: https://physionet.org/content/mimic-cxr/2.1.0/
- **Pitfalls:**
    - It comes from one hospital's emergency department.
    - Most studies have more than one image (frontal plus lateral), so split on `subject_id`, not on image or study.

## MIMIC-CXR-JPG

- **Name and maintainer:**
    - MIMIC-CXR-JPG v2.1.0, from the MIT Laboratory for Computational Physiology on PhysioNet.
    - Source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- **Tasks and labels:**
    - Multi-label classification, with no boxes or masks.
    - 14 labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumonia, Pneumothorax, Pleural Other, Support Devices, and No Finding.
    - The labels are **NLP-derived** from the reports by two labelers, CheXpert and NegBio.
    - Values are 1.0 (positive), 0.0 (negative), -1.0 (uncertain), and blank (not mentioned).
    - A radiologist-labeled test set is in `mimic-cxr-2.1.0-test-set-labeled.csv`.
    - Source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- **Label format:**
    - Gzipped CSV: `mimic-cxr-2.0.0-chexpert.csv.gz`, `mimic-cxr-2.0.0-negbio.csv.gz`, and the official split `mimic-cxr-2.0.0-split.csv.gz` with columns `dicom_id, study_id, subject_id, split` (train/validate/test).
    - A metadata CSV has `Rows` and `Columns`.
    - Source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- **Size:**
    - 377,110 images and 227,827 labeled studies (https://physionet.org/content/mimic-cxr-jpg/2.1.0/, https://arxiv.org/abs/1901.07042).
    - The patient count is not stated on the PhysioNet JPG page. It is the same image set as MIMIC-CXR, which has 65,379 patients.
    - **Patient IDs (`subject_id`) are available**, and the official split is at the patient level.
- **License and access:**
    - PhysioNet Credentialed Health Data License 1.5.0, with CITI "Data or Specimens Only Research" training and the PhysioNet Credentialed Health Data Use Agreement 1.5.0 (https://physionet.org/content/mimic-cxr-jpg/2.1.0/).
    - Access is **credentialed**.
- **Image format and resolution:**
    - JPEG (quality 95) converted from DICOM.
    - Pixel values are normalized to 0 to 255 with histogram equalization.
    - There is no fixed resolution; per-image `Rows` and `Columns` are in the metadata.
    - Source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- **Citation:** Johnson AEW, Pollard TJ, Greenbaum NR, Lungren MP, Deng C-y, Peng Y, Lu Z, Mark RG, Berkowitz SJ, Horng S. "MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs." arXiv:1901.07042, 2019 (https://arxiv.org/abs/1901.07042). Dataset DOI (v2.1.0): 10.13026/jsn5-t979.
- **Pitfalls:**
    - The two NLP labelers disagree, and uncertain labels are frequent.
    - The histogram equalization differs from CheXpert preprocessing, so the two datasets are not directly comparable.
    - The page reports 227,827 labeled studies against 227,943 reports and does not explain the gap.

## NIH ChestX-ray14 (ChestX-ray8)

- **Name and maintainer:**
    - NIH Chest X-ray dataset (ChestX-ray14, an expansion of ChestX-ray8), from the NIH Clinical Center.
    - The official download is on Box: https://nihcc.app.box.com/v/ChestXray-NIHCC. The Box page is rendered by JavaScript, and its README could not be fetched.
- **Tasks and labels:**
    - Multi-label classification over 14 findings plus "No Finding": Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia.
    - The labels are **NLP text-mined from reports**, with expected accuracy above 90%.
    - A small **bounding-box** set: radiologists hand-labeled 1,600 instances (200 for each of the 8 original ChestX-ray8 findings) on 983 images. This set was unchanged for ChestX-ray14.
    - Sources: https://arxiv.org/abs/1705.02315 (v5 appendix B) and https://www.kaggle.com/datasets/nih-chest-xrays/data
    - Google has released adjudicated expert labels for subsets: 4 findings on 4,374 images, and all findings on 810 test images (https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest).
- **Label format:**
    - `Data_Entry_2017.csv`: Image Index, Finding Labels (pipe-separated), Follow-up #, Patient ID, Patient Age, Patient Gender, View Position, and original width, height, and pixel spacing.
    - `BBox_List_2017.csv`: Image Index, Finding Label, x, y, w, and h.
    - Source: Kaggle mirror description, https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Size:**
    - 112,120 frontal-view images (arXiv v5, https://arxiv.org/abs/1705.02315) from 30,805 unique patients (Kaggle NIH page, https://www.kaggle.com/datasets/nih-chest-xrays/data).
    - **Patient IDs are available** in the `Patient ID` column.
    - The authors' own benchmark split is at the patient level (arXiv v5).
- **License and access:**
    - "There are no restrictions on the use of the NIH chest x-ray images."
    - Attribution is required: link to the NIH download site, cite the CVPR 2017 paper, and acknowledge the NIH Clinical Center (Google Cloud mirror, https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest).
    - The Kaggle mirror is tagged "CC0: Public Domain" (https://www.kaggle.com/datasets/nih-chest-xrays/data). That tag was set on the Kaggle upload; the NIH Box README itself could not be read.
    - NIH states no named license on the sources fetched.
    - Access is **open**.
- **Image format and resolution:**
    - PNG, 1024x1024, resized from DICOM and rescaled with the DICOM default window (https://arxiv.org/abs/1705.02315).
    - DICOM versions are on Google Cloud.
- **Citation:** Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." IEEE CVPR 2017, pp. 3462-3471. arXiv:1705.02315 (https://arxiv.org/abs/1705.02315). The page range differs by source: arXiv lists pp. 2097-2106, while NIH, Google, and RSNA attribution text gives pp. 3462-3471.
- **Pitfalls:**
    - The NLP labels are noisy, the reports were never released, and there are very few boxes.
    - Many patients have many follow-up images (the `Follow-up #` column), so splits by image leak patients. Use the patient-level split.

## RSNA Pneumonia Detection Challenge (2018)

- **Name and maintainer:**
    - RSNA Pneumonia Detection Challenge, by RSNA with the Society of Thoracic Radiology (STR) and MD.ai. It ran on Kaggle.
    - The images are a subset of NIH ChestX-ray.
    - Sources: https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018 and https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
- **Tasks and labels:**
    - **Detection with bounding boxes** of pneumonia-like opacity, plus image-level classes.
    - **Radiologist-annotated** by 18 board-certified radiologists from 16 institutions, 12 of them STR thoracic radiologists.
    - Readers chose Opacity (high, medium, or low probability), "No Opacity/Not Normal", or "Normal". The adjudicated challenge version uses a single box category plus "No Opacity/Not Normal".
    - 4,527 cases were triple-read.
    - Source: Shih et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC8017407/
- **Label format:**
    - JSON annotation files on the RSNA site, original and adjudicated, plus a JSON mapping to the original NIH image IDs (https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).
    - The Kaggle CSV layout (`patientId, x, y, width, height, Target`) is **not verified**, because the Kaggle page could not be read.
- **Size:**
    - 30,000 frontal radiographs: 16,248 PA and 13,752 AP.
    - The test set is 4,527 per the RSNA dataset description, while the paper describes a 90:10 split with 3,000 test cases.
    - 12,274 unique patients (https://pmc.ncbi.nlm.nih.gov/articles/PMC8017407/).
    - The RSNA mapping file links images back to NIH image IDs, and so to NIH Patient IDs. Whether the RSNA files themselves expose a patient ID is not stated on the RSNA page.
- **License and access:**
    - RSNA's "Pneumonia Detection Challenge Terms of Use and Attribution" allows use "for the purposes of academic research and education, and other commercial or non-commercial purposes".
    - Conditions: no attempt to identify subjects, and if the data is redistributed, the NIH attribution (Box link, CVPR 2017 citation, NIH Clinical Center) plus the RSNA attribution (RSNA download link, Shih et al. 2019 citation).
    - Source: https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf
    - No named license (such as CC BY) is given. The Kaggle competition rules were not verified.
    - Access is **open**, through direct ZIP and JSON downloads from RSNA S3 links. The Kaggle copy needs a Kaggle account and acceptance of the competition rules.
- **Image format and resolution:**
    - DICOM, converted from the NIH PNGs, with sex, age, and projection added to the tags (RSNA dataset description PDF, linked from the RSNA page).
    - The resolution is not stated on the RSNA page or the paper. The source NIH PNGs are 1024x1024.
- **Citation:** Shih G, Wu CC, Halabi SS, Kohli MD, Prevedello LM, Cook TS, Sharma A, Amorosa JK, Arteaga V, Galperin-Aizenberg M, Gill RR, Godoy MCB, Hobbs S, Jeudy J, Laroia A, Shah PN, Vummidi D, Yaddanapudi K, Stein A. "Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia." *Radiology: Artificial Intelligence* 1(1):e180041, 2019. doi:10.1148/ryai.2019180041
- **Pitfalls:**
    - The images are a subset of NIH ChestX-ray14, so models pretrained on NIH may already have seen the RSNA test images. Map through the RSNA-to-NIH JSON and split by NIH Patient ID.
    - The test-set size differs between RSNA's description (4,527) and the paper (3,000).

## VinDr-CXR

- **Name and maintainer:**
    - VinDr-CXR v1.0.0, from Vingroup Big Data Institute (VinBigData), Vietnam, hosted on PhysioNet.
    - The images come from Hospital 108 and Hanoi Medical University Hospital, 2018 to 2020.
    - Sources: https://physionet.org/content/vindr-cxr/1.0.0/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC9300612/
- **Tasks and labels:**
    - **Detection with bounding boxes** for 22 local findings, plus 6 image-level global labels.
    - Local findings: aortic enlargement, atelectasis, cardiomegaly, calcification, clavicle fracture, consolidation, edema, emphysema, enlarged PA, ILD, infiltration, lung cavity, lung cyst, lung opacity, mediastinal shift, nodule/mass, pulmonary fibrosis, pneumothorax, pleural thickening, pleural effusion, rib fracture, and other lesion.
    - Global labels: lung tumor, pneumonia, tuberculosis, other diseases, COPD, and no finding.
    - All labels are **radiologist-annotated**, by 17 radiologists in total. Each training image was labeled independently by 3 radiologists, and each test label is the consensus of 5.
    - Source: https://physionet.org/content/vindr-cxr/1.0.0/
- **Label format:**
    - `annotations_{train,test}.csv`: `image_id, class_name, x_min, y_min, x_max, y_max`, and the training file adds `rad_id`.
    - `image_labels_{train,test}.csv`: the 0/1 global labels, with `rad_ID` in the training file.
    - Training boxes are **per radiologist and not merged**.
    - Source: https://physionet.org/content/vindr-cxr/1.0.0/
- **Size:**
    - 18,000 PA-view adult images: 15,000 for training (10,606 normal) and 3,000 for test (2,052 normal).
    - **No patient or study ID** is provided. `image_id` is a hash of the SOP Instance UID, and the paper reports scan counts, not patient counts.
    - Sources: https://physionet.org/content/vindr-cxr/1.0.0/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC9300612/
- **License and access:**
    - PhysioNet Credentialed Health Data License 1.5.0, with CITI "Data or Specimens Only Research" training and the PhysioNet Credentialed Health Data Use Agreement 1.5.0 (https://physionet.org/content/vindr-cxr/1.0.0/).
    - Access is **credentialed**.
- **Image format and resolution:**
    - DICOM.
    - Median size is 2788x2446 (training) and 2748x2394 (test). Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC9300612/
- **Citation:** Nguyen HQ, Lam K, Le LT, et al. (25 authors). "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations." *Scientific Data*, 2022. doi:10.1038/s41597-022-01498-w. Preprint arXiv:2012.15029. Dataset DOI: 10.13026/3akn-b287. Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC9300612/ and https://physionet.org/content/vindr-cxr/1.0.0/
- **Pitfalls:**
    - Patient-level splitting is impossible because there are no patient IDs.
    - Some classes are very rare: edema has 1 training case and 0 test cases.
    - Training labels from 3 independent readers and test labels from a 5-reader consensus are not comparable.
    - All images come from two Vietnamese hospitals.

## PadChest

- **Name and maintainer:**
    - PadChest, from BIMCV (Medical Imaging Databank of the Valencia Region) and the University of Alicante.
    - The images come from Hospital San Juan, Alicante, Spain, 2009 to 2017.
    - Source: https://bimcv.cipf.es/bimcv-projects/padchest/
- **Tasks and labels:**
    - Multi-label classification over a hierarchical, UMLS-mapped taxonomy: 174 radiographic findings, 19 differential diagnoses, and 104 anatomical locations. There are no boxes or masks.
    - **About 27% of reports were labeled manually by physicians. The rest were labeled by an RNN-with-attention model** (micro-F1 0.93 on a held-out set).
    - The reports are in Spanish.
    - Sources: https://bimcv.cipf.es/bimcv-projects/padchest/ and https://arxiv.org/abs/1901.07441
- **Label format:**
    - One CSV with 33 fields per study, including `ImageID`, `StudyID`, `PatientID`, and `ReportID`.
    - Labels are stored as lists in fields. Fields with a `_DICOM` suffix hold the original DICOM tags.
    - The paper does not give the CSV filename. Source: https://arxiv.org/abs/1901.07441
- **Size:**
    - 160,868 images, 109,931 studies, and 67,625 patients (paper, final labeled set).
    - The abstract says "more than 160,000" images from about 67,000 patients, and the paper's figures differ by section.
    - **Patient IDs are available** (`PatientID`). Source: https://arxiv.org/abs/1901.07441
- **License and access:**
    - The PADCHEST Dataset Research Use Agreement. Use is free for researchers.
    - The user commits not to share the data and not to re-identify anyone.
    - The full dataset (about 1 TB) requires a formal request through a Google Form (https://bimcv.cipf.es/bimcv-projects/padchest/).
    - No standard license name is stated on the BIMCV page or the paper, and the BIMCV page states no commercial-use terms.
    - A 167 MB sample is on Kaggle.
    - Access is **registration**.
- **Image format and resolution:**
    - PNG converted from DICOM (the `ImageID` values end in `.png`), with DICOM `BitsStored` of 12 in the example.
    - Images were not resized: the mean is about 2254 x 2299 px, at 0.1 to 0.2 mm pixel spacing. Source: https://arxiv.org/abs/1901.07441
    - The PNG bit depth is not stated on the BIMCV page or the paper.
- **Citation:** Bustos A, Pertusa A, Salinas J-M, de la Iglesia-Vayá M. "PadChest: A large chest x-ray image dataset with multi-label annotated reports." *Medical Image Analysis* 66:101797, 2020. doi:10.1016/j.media.2020.101797. arXiv:1901.07441
- **Pitfalls:**
    - About 73% of the labels are model-generated, and "unchanged" labels refer to prior studies.
    - Six view types are mixed together, including pediatric images.
    - Everything comes from a single site.
    - The paper's label counts are internally inconsistent (174, 170, or 193 findings).

## SIIM-ACR Pneumothorax Segmentation (2019)

- **Name and maintainer:**
    - SIIM-ACR Pneumothorax Segmentation, by the Society for Imaging Informatics in Medicine and the American College of Radiology, with STR and MD.ai. It ran on Kaggle.
    - The images are from NIH ChestX-ray.
    - Sources: https://siim.org/page/pneumothorax_challenge and https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation
- **Tasks and labels:**
    - **Segmentation** of pneumothorax, with an implied image-level classification.
    - Radiologists drew the masks. Six radiologists did the initial freeform segmentation, 10,902 manually and 4,355 by reviewing Mask R-CNN proposals.
    - Thirteen STR thoracic radiologists did the review and adjudication.
    - Source: Filice et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC7165201/
- **Label format:**
    - Run-length-encoded masks keyed by ImageId, with "-1" meaning no pneumothorax.
    - **Not verified on a primary page**: the Kaggle data page could not be read, and only secondary sources report the format.
- **Size:**
    - 15,257 annotated exams (https://pmc.ncbi.nlm.nih.gov/articles/PMC7165201/).
    - The Kaggle release counts and the number of positive cases are not stated on the SIIM page or the paper.
    - Whether patient IDs are exposed is not stated on the sources checked.
- **License and access:**
    - The license is not stated on https://siim.org/page/pneumothorax_challenge or in the paper.
    - The Kaggle competition rules were not verified.
    - Access is **registration** (a Kaggle account and rules acceptance; inferred from Kaggle hosting, not read on the page).
- **Image format and resolution:**
    - DICOM, converted from the NIH PNGs for annotation (https://pmc.ncbi.nlm.nih.gov/articles/PMC7165201/).
    - The resolution is not stated on the sources checked. The source NIH PNGs are 1024x1024.
- **Citation:** Filice RW, et al. (20 authors). "Crowdsourcing pneumothorax annotations using machine learning annotations on the NIH chest X-ray dataset." *Journal of Digital Imaging* 33(2):490-496, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7165201/. The DOI was not read on the publisher page.
- **Pitfalls:**
    - The images are a subset of NIH ChestX-ray14, so the same leakage risk applies as for RSNA.
    - The paper notes that chest tubes were annotated separately; tubes in treated pneumothorax cases are a known shortcut cue.
    - Check the RLE convention before decoding.

---

## Unverified

- **CheXpert:**
    - The exact title of the Research Use Agreement. TFDS calls it a "Research Use Agreement", and the AIMI page shows empty license headings.
    - The full-resolution JPEG pixel size.
- **CheXpert Plus:** image resolution and RadGraph file format.
- **MIMIC-CXR:** DICOM resolution and bit depth.
- **NIH ChestX-ray14:**
    - The official NIH Box README could not be fetched, because Box renders it with JavaScript. The license wording comes from the Google Cloud mirror and the CC0 tag from the Kaggle mirror.
    - The 30,805 patient count comes from the Kaggle NIH-owned page, not the paper.
    - The commonly quoted "984 boxes" was not verified: the paper says 1,600 instances on 983 images.
    - The CVPR page range is inconsistent between sources.
- **RSNA:**
    - The Kaggle CSV schema, Kaggle competition rules, and image resolution.
    - The test-set size conflicts between sources (4,527 against 3,000).
    - The RSNA dataset-description PDF has a "Data use agreement/licensing" checklist ("Open licensing", "Non-commercial purpose"), but which boxes are ticked could not be read from the PDF text. The separate Terms of Use PDF explicitly allows commercial use.
- **VinDr-CXR:** the volume and article number, and the author list beyond the first three. The PMC page confirms the title, journal, year, DOI, and 25 authors.
- **SIIM-ACR:**
    - The Kaggle data format (RLE CSV), image counts, license and rules, and resolution. None of these could be read from Kaggle.
    - The paper's author list beyond the first author, and its DOI.
- **PadChest:** PNG bit depth, the CSV filename, and commercial-use terms. The PADCHEST Research Use Agreement text itself was not read.
- **Extra datasets** (Object-CXR, TBX11K): not added, because they were not verified.
