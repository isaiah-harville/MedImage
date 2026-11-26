# MRI and CT datasets: details

Part of the [dataset survey](../datasets.md).

Checked 2026-11-26 against primary sources (official dataset pages, Synapse wiki via the Synapse REST API, TCIA collection pages, Zenodo records, GitHub READMEs, and the dataset papers via PMC, arXiv and Crossref). Where a primary source does not state something, the entry says so instead of filling it in. **UNVERIFIED** marks a claim I could not confirm on a primary source.

Access classes: **open** (direct download, at most a click-through license), **registration** (account and terms acceptance, no human review), **credentialed** (application or DUA reviewed by the provider).

## MRI

### BraTS (Brain TumorS), 2026 edition and lineage

- **Name and maintainer:** BraTS 2026, "The 15th BraTS Cluster of Challenges at MICCAI 2026" (Synapse project syn74274097). The acronym is now "Brain TumorS" (formerly "Brain Tumor Segmentation"). Organizers include S. Bakas (Indiana University) and others; it is hosted on Synapse by Sage Bionetworks. Source: https://www.synapse.org/Synapse:syn74274097/wiki/639571
- **Tasks and labels:**
  - 2026 has five tasks: (1) pre- and post-treatment brain metastases segmentation (METS), (2) pediatric tumor segmentation (PEDS), (3) generalizability across tumors (GoAT), (4) MR local inpainting (synthesis), and (5) glioma histopathology classification (BraTS-Path). Source: the 2026 wiki tree, https://www.synapse.org/Synapse:syn74274097/wiki/639603
  - 2025 (BraTS-Lighthouse, syn64153130) had 11 tasks across segmentation, synthesis and classification. Source: https://www.synapse.org/Synapse:syn64153130/wiki/630130
  - Glioma label map (2025 task 1): ET = 3 (enhancing tissue), SNFH = 2 (surrounding non-enhancing FLAIR hyperintensity), NETC = 1 (non-enhancing tumor core), RC = 4 (resection cavity). Evaluated regions include TC = ET+NETC and WT = ET+SNFH+NETC. Source: https://www.synapse.org/Synapse:syn64153130/wiki/631053
  - BraTS 2021 also had a radiogenomic classification task (MGMT promoter methylation). Source: arXiv:2107.02314.
- **Label format:** NIfTI voxel masks. The 2025 page states "All BraTS mpMRI scans are available as NIfTI files (.nii.gz)". Source: https://www.synapse.org/Synapse:syn64153130/wiki/631053
- **Size:**
  - 2026 METS: 1,296 train, 179 validation and 303 test cases from 8 institutions, plus 245 unannotated Ulm cases. Source: https://www.synapse.org/Synapse:syn74274097/wiki/639600
  - BraTS 2021: "pre-operative baseline mpMRI data from 2,040 patients". Source: arXiv:2107.02314 abstract.
  - Per-task patient counts for the other 2026 tasks are not stated on the data pages I read.
  - Patient IDs: yes. Names follow `BraTS-GLI-<5-digit subject ID>-<3-digit timepoint>`; for example 000 is the first pre-operative scan and 100 the first post-treatment scan. Group longitudinal timepoints by subject ID for patient-level splits. Source: https://www.synapse.org/Synapse:syn64153130/wiki/631053
- **License and access:**
  - License: "BraTS data is subject to a CC BY-NC license, unless otherwise indicated" (linked to CC BY-NC 4.0). The same clause appears in the 2025 and 2026 rules. Sources: https://www.synapse.org/Synapse:syn74274097/wiki/639585 and https://www.synapse.org/Synapse:syn64153130/wiki/631064
  - Terms: users must follow the Sage Bionetworks Terms of Service and cite the challenge papers.
  - Since 2026, downloaders must also agree to comply with EO 14117, 28 CFR Part 202 and NIH NOT-OD-25-083. Source: https://www.synapse.org/Synapse:syn74274097/wiki/639585
  - Access steps: a Synapse account, challenge registration, a Google data-access form, then joining the data-access team.
  - The 2026 data pages say data access "is temporarily not available to non-registered participants". The 2025 page says the post-challenge dataset "will be hosted on The Cancer Imaging Archive (TCIA)". Source: https://www.synapse.org/Synapse:syn64153130/wiki/631048
  - Validation ground truth and test sets are never released.
  - **Access class: registration.**
- **Image format and dimensionality:** 3D NIfTI, with four co-registered sequences per case (T1, T1Gd, T2, FLAIR).
  - Adult glioma data are co-registered to one anatomical template, resampled to 1 mm³ and skull-stripped (2025 task 1).
  - 2026 METS mixes native-space and SRI24-space cases.
  - PEDS images are defaced but not skull-stripped.
- **Primary citation:**
  - Menze BH et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE TMI 34(10), 2015, doi:10.1109/TMI.2014.2377694.
  - Baid U et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314 (2021).
  - Post-treatment glioma: de Verdier MC et al., arXiv:2405.18368 (2024).
  - Links: https://www.synapse.org/Synapse:syn74274097 (2026), https://www.synapse.org/Synapse:syn64153130 (2025).
- **Pitfalls:**
  - The rules forbid training on other BraTS sub-challenges, and GoAT forbids all external data, so mixing editions can disqualify a submission or leak test cases.
  - The MSD "BrainTumour" task reuses BraTS 2016/2017 cases under renamed files (Antonelli 2022).
  - Labels and the label map changed across editions (for example RC was added in 2024).
  - Data are multi-site and multi-scanner, and 2025/2026 METS mixes native and SRI24 space.

### fastMRI (knee, brain, prostate, breast)

- **Name and maintainer:** fastMRI, NYU Langone Health (Center for Advanced Imaging Innovation and Research), originally with Facebook AI Research. The facebookresearch/fastMRI GitHub repository is archived (2025-08-18). Source: https://fastmri.med.nyu.edu/
- **Tasks and labels:**
  - Main task: reconstruction (accelerated MRI from undersampled k-space).
  - Knee and brain ship no pathology labels. fastMRI+ (a separate project) adds bounding boxes and study-level labels: 16,154 knee boxes in 22 pathology categories, and 7,570 boxes plus 643 study-level labels across 30 brain categories. Source: arXiv:2109.03812
  - Prostate: slice-, volume- and exam-level labels for cancer presence and grade. Sources: arXiv:2304.09254 and https://github.com/cai2r/fastMRI_prostate
  - Breast: case-level age, menopause status, lesion status (negative, benign or malignant) and lesion type. Source: arXiv:2406.05270
- **Label format:**
  - Reconstruction targets are inside the HDF5 files: `reconstruction_rss`, plus `reconstruction_esc` for single-coil. Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md
  - fastMRI+ labels are CSV (`knee.csv`, `brain.csv`, one row per bounding box). Source: https://github.com/microsoft/fastmri-plus
  - Prostate and breast label file formats: not stated on the pages read.
- **Size:**
  - Knee: the official page says "over 1,500 fully sampled raw knee MRIs" (its breakdown lists 1,398 PD/PDFS) plus DICOM from 10,000 clinical knee MRIs. The Knoll 2020 paper gives 1,594 raw k-space datasets and 10,012 DICOM datasets from 9,290 patients. Sources: https://fastmri.med.nyu.edu/ and PMC6996599
  - Brain: the page says 6,970 raw brain MRIs, while its table totals 7,002. Muckley 2021 says 7,299 scans, with 894 held back. Source: PMC8428775
  - Prostate: 312 exams. Breast: 300 exams. Source: https://fastmri.med.nyu.edu/
  - Patient IDs: not stated on the pages read. Files are per acquisition, so a patient-level split cannot be confirmed from public metadata.
- **License and access:**
  - "NYU Langone Health fastMRI Dataset Sharing Agreement". The page says each dataset has its own agreement. Source: https://fastmri.med.nyu.edu/
  - Terms: internal research or educational use only; no selling or monetizing; no sharing the download link; citation required.
  - Access: accept the agreement and submit an application with institutional affiliation and proposed uses; a download link follows approval.
  - The code is MIT; this is not a data license. Source: https://github.com/facebookresearch/fastMRI
  - fastMRI+ annotations: "MIT license". Source: https://github.com/microsoft/fastmri-plus
  - **Access class: credentialed** (application approved by NYU).
- **Image format and dimensionality:**
  - Raw data: HDF5, one file per acquisition. `kspace` has shape (slices, H, W) for single-coil or (slices, coils, H, W) for multi-coil; test files have `mask`. Source: the fastMRI data README above.
  - The official page instead describes raw data as "vendor-neutral ISMRMRD format".
  - Clinical images are DICOM.
  - Data are 2D multi-slice per volume. Breast is 3D radial GRASP DCE.
- **Primary citation:**
  - Zbontar J et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI", arXiv:1811.08839 (2018).
  - Knoll F et al., "fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images…", Radiology: Artificial Intelligence 2(1), 2020, doi:10.1148/ryai.2020190007.
  - Tibrewala R et al., "FastMRI Prostate…", Scientific Data 11, 2024, doi:10.1038/s41597-024-03252-w.
  - Solomon E et al., "FastMRI Breast: A Publicly Available Radial k-Space Dataset of Breast DCE MRI", Radiology: Artificial Intelligence 7, 2025, doi:10.1148/ryai.240345.
- **Pitfalls:**
  - Single-coil knee data is emulated from multi-coil, not acquired.
  - Train and validation volumes are shared between the single-coil and multi-coil tracks.
  - Slices of one volume are highly correlated, so split by volume (ideally by patient).
  - fastMRI+ pixel arrays were flipped up/down during DICOM conversion, and each image was read by only one radiologist.

### OASIS (OASIS-1, OASIS-2, OASIS-3; OASIS-4 noted)

- **Name and maintainer:** Open Access Series of Imaging Studies, Washington University in St. Louis (Knight ADRC). Data are hosted on NITRC-IR. Source: https://sites.wustl.edu/oasisbrains/
- **Tasks and labels:**
  - Classification and regression: dementia status, cognitively normal vs cognitive decline, age.
  - Segmentation: FreeSurfer outputs are derived, not manual labels.
  - OASIS-2 subjects are characterized by Clinical Dementia Rating (CDR). Source: Marcus 2010
  - OASIS-3 includes UDS clinical forms, cognitive assessments, PET Centiloid values and APOE status. Sources: https://sites.wustl.edu/oasisbrains/home/oasis-3/ and LaMontagne 2019
- **Label format:** CSV and spreadsheets for demographic and clinical data (an OASIS-1 CSV; OASIS-3 FreeSurfer and PUP output spreadsheets). Source: https://sites.wustl.edu/oasisbrains/home/oasis-1/
- **Size:**
  - OASIS-1: 416 subjects (ages 18–96), 434 MR sessions, 3–4 T1w scans per session. 100 subjects have very mild to moderate AD; 20 nondemented subjects form a reliability rescan set.
  - OASIS-2: 150 subjects (ages 60–96), 373 sessions.
  - OASIS-3: 1,378 participants on the site summary (1,379 in its count box; the 2019 preprint says 1,098), with 2,842 MR, 2,157 PET and 1,472 CT sessions.
  - OASIS-4: 663 subjects.
  - Sources: https://sites.wustl.edu/oasisbrains/ and the OASIS-1 and OASIS-3 pages.
  - Patient IDs: yes. "All participants were assigned a new random identifier", and dates are given as days from entry. Source: OASIS-3 page.
- **License and access:**
  - No named license. The site points to a "Data Use Agreement" (July 2025 PDF) whose terms include "OASIS data are provided for non-commercial, academic research purposes only".
  - Other terms: no re-identification, no facial reconstruction, secure storage, acknowledgment in publications, and Avid review 30 days before publishing AV45/AV1451 PET work. Source: https://sites.wustl.edu/oasisbrains/
  - Access: a request through the site, reviewed by OASIS staff, then granted on NITRC. Tau PET needs a separate request.
  - **Access class: credentialed.**
- **Image format and dimensionality:** 3D MRI (T1w for OASIS-1 and -2; OASIS-3 adds T2w, FLAIR, ASL, SWI, TOF, BOLD, DTI and PET).
  - On-disk format (Analyze, NIfTI, DICOM or BIDS): **not stated on the pages read.** OASIS-1 is distributed as 12 .tar.gz archives.
- **Primary citation:**
  - Marcus DS et al., "Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data…", J Cogn Neurosci 19(9), 2007, doi:10.1162/jocn.2007.19.9.1498.
  - Marcus DS et al., "…longitudinal MRI data in nondemented and demented older adults", J Cogn Neurosci 22, 2010, doi:10.1162/jocn.2009.21407.
  - LaMontagne PJ et al., "OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset…", medRxiv 2019, doi:10.1101/2019.12.13.19014902.
- **Pitfalls:**
  - Multiple scans per session (OASIS-1/2) and longitudinal visits (OASIS-2/3) leak across splits unless you split by subject.
  - The OASIS-1 reliability rescans duplicate subjects.
  - OASIS-3 counts differ between the paper and the website.

### ADNI (Alzheimer's Disease Neuroimaging Initiative)

- **Name and maintainer:** ADNI, PI Michael W. Weiner. Data are disseminated by the Laboratory of Neuro Imaging (LONI), USC, through the Image and Data Archive (IDA). Source: the ADNI DUA text (Rev. 2016-12-20), https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Data_Use_Agreement.pdf
- **Tasks and labels:**
  - Classification and regression: diagnosis groups CN / MCI (LMCI in some tables) / dementia, plus progression and conversion.
  - Data types: MRI, PET, clinical assessments, biofluid biomarkers, genetics and neuropathology. Source: https://adni.loni.usc.edu/data-samples
- **Label format:** Tabular clinical data. The specific file format (CSV) is **not stated on the pages read.**
- **Size:** 2,470 participants across ADNI-1 (819), ADNI-GO (131), ADNI-2 (790), ADNI-3 (692) and ADNI-4 (38 as of 2024-06-17).
  - Source: Beckett LA et al., "Contributions of the ADNI Biostatistics Core", Alzheimer's & Dementia 2024, doi:10.1002/alz.14159 (PMC11485306).
  - The official ADNI site does not state a total on the pages read.
  - Patient IDs: longitudinal per-participant data, but the RID/PTID scheme is **not verified** on a primary page.
- **License and access:**
  - "Alzheimer's Disease Neuroimaging Initiative (ADNI) Data Use Agreement". It states "redistribution of data in any manner is prohibited".
  - Terms: annual update requests; required by-line and acknowledgment text; all manuscripts submitted to the ADNI Data and Publications Committee before journal submission; no attempt to identify subjects. Source: https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Data_Use_Agreement.pdf (the version served is Rev. December 20, 2016)
  - Access: an online application through LONI IDA. https://adni.loni.usc.edu/data-samples links "Access Data through IDA".
  - The same page notes: "This repository is under review for potential modification in compliance with Administration directives."
  - **Access class: credentialed** (application reviewed).
- **Image format and dimensionality:** 3D MRI. The protocol includes back-to-back 3D MP-RAGE and axial PD/T2 FSE (Jack 2008). The download format (DICOM or NIfTI) is **not stated on the pages read.**
- **Primary citation:** Jack CR Jr et al., "The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods", J Magn Reson Imaging 27, 2008, doi:10.1002/jmri.21049. Official page: https://adni.loni.usc.edu/
- **Pitfalls:**
  - Many scanners and sites, and protocol changes across phases.
  - Repeated MP-RAGE scans and longitudinal visits per subject are a classic source of leakage in AD-classification papers; split by participant.
  - Processed data may be preliminary, and the DUA asks users to record the data version they downloaded.

### UK Biobank imaging

- **Name and maintainer:** UK Biobank (UK charity), imaging enhancement study.
- **Tasks and labels:**
  - Population cohort, not a curated benchmark. Uses include image-derived phenotypes (IDPs), regression, and linkage to health outcomes.
  - Modalities: brain, cardiac and abdominal MRI, DXA, carotid ultrasound, and a 12-lead ECG. Source: Littlejohns 2020 (PMC7250878)
- **Label format:** IDPs and data fields are selected through the Access Management System. The file format is **not verified.**
- **Size:**
  - Target: 100,000 imaged participants of roughly 500,000, planned to finish by end-2023. Over 45,000 to almost 50,000 had been imaged by early 2020, and at least 10,000 are invited for repeat imaging. Source: Littlejohns 2020
  - The current imaged count is **not verified** (the ukbiobank.ac.uk pages are behind a bot-check and were not fetched).
  - Participant IDs: project-specific encoded identifiers ("EID"), "algorithmically generated by UK Biobank specifically for a particular research application". Source: UK Biobank Access Procedures (PDF), §B13.6, https://www.ukbiobank.ac.uk/wp-content/uploads/2026/05/Access-procedures.pdf
- **License and access:**
  - No open license. Under the "UK Biobank Access Procedures" the Resource "is available to all bona fide researchers for all types of health-related research that is in the public interest".
  - Access needs an approved application, a signed "Material Transfer Agreement (MTA)" per institution, and an access fee. Researchers must return results to UK Biobank. Source: the Access Procedures PDF above (titled "July 2022 v2.1").
  - Access runs through the UK Biobank Research Analysis Platform (UKB-RAP).
  - Current fee amounts and RAP status: **not verified** (a search snippet from the fees page quoted £3,000 for three years, but the page was not fetched).
  - **Access class: credentialed** (plus a fee).
- **Image format and dimensionality:** Brain T1 and T2 are NIfTI (defaced); DXA images are DICOM; both are 3D. Source: Littlejohns 2020
- **Primary citation:** Littlejohns TJ et al., "The UK Biobank imaging enhancement of 100,000 participants: rationale, data collection, management and future directions", Nature Communications 11, 2020, doi:10.1038/s41467-020-15948-9.
- **Pitfalls:**
  - EIDs differ between applications, so they cannot be joined across projects.
  - Repeat-imaging visits must stay within one split.
  - The cohort shows healthy-volunteer selection bias.

### Duke Breast Cancer MRI (TCIA)

- **Name and maintainer:** Duke-Breast-Cancer-MRI, from Duke University (Mazurowski lab), hosted by TCIA. Source: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/
- **Tasks and labels:**
  - Detection: 3D tumor bounding boxes (start/end row, column and slice) drawn by radiologists.
  - Segmentation: breast and fibroglandular tissue masks.
  - Classification and radiogenomics: clinical features (demographics, tumor characteristics, treatment, recurrence, follow-up) and 529 computer-extracted imaging features.
- **Label format:** XLSX (`Annotation Boxes`, `Clinical_and_Other_Features.xlsx`, `Imaging_Features.xlsx`). Segmentations come as NRRD, TXT and DICOM SEG.
- **Size:** 922 patients, 922 studies, 5,161 series and 773,253 images (version 3, 2022-08-01). Patients have IDs, one study each.
- **License and access:** "CC BY-NC 4.0" per the TCIA collection page. Images download through the TCIA Data Retriever, and the collection is also listed on the NCI Imaging Data Commons. Source: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/
  - **Access class: open.** The page does not label an access type, and no application is described.
- **Image format and dimensionality:** DICOM, 3D DCE-MRI (pre-contrast plus post-contrast phases).
- **Primary citation:**
  - Saha A et al., "A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 DCE-MRI features", British Journal of Cancer 119(4):508–516, 2018, doi:10.1038/s41416-018-0185-8.
  - Data citation: doi:10.7937/TCIA.e3sv-re93.
- **Pitfalls:**
  - Single institution, collected 2000–2014, so expect scanner and protocol drift over time.
  - The boxes cover the largest biopsied lesion or the mass/non-mass enhancement, not every lesion.

### ACDC (Automated Cardiac Diagnosis Challenge)

- **Name and maintainer:** ACDC, CREATIS (Lyon) and University Hospital of Dijon, from the MICCAI 2017 challenge. Sources: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html and https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb
- **Tasks and labels:**
  - Segmentation of LV cavity, LV myocardium and RV at end-diastole and end-systole frames.
  - Classification into 5 groups: NOR, MINF, DCM, HCM, RV.
  - Per-patient metadata: weight, height and ED/ES frame indices (the training page also lists age).
  - Integer label encoding: **not stated on the pages read.**
- **Label format:** NIfTI masks (`patientXXX_frameYY_gt.nii.gz`) plus an `Info.cfg` text file per patient. Source: Girder file listing of `training/patient001`.
- **Size:** 150 exams from 150 different patients (100 training with 20 per group; 50 testing). The patient folders count 100 and 50, and each folder is one patient.
- **License and access:** "CC BY-NC-SA 4.0" plus terms that it is free "strictly only" for non-commercial scientific research and that the Bernard 2018 citation is mandatory. Source: `LICENSE_TERMS.md` in the official Girder collection, https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb
  - The collection is flagged `"public": true` in the Girder API.
  - **Access class: open.**
- **Image format and dimensionality:** NIfTI. Cine short-axis volumes (`_4d.nii.gz`, 3D+time) plus 3D ED/ES frames. Slices are 5–8 mm thick; acquired on 1.5 T and 3 T Siemens scanners.
- **Primary citation:** Bernard O et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?", IEEE TMI 37(11):2514–2525, 2018, doi:10.1109/TMI.2018.2837502.
- **Pitfalls:**
  - Single center, small, and balanced by design, so the class priors are not clinical.
  - Thick, anisotropic slices.
  - The training page says one clinical expert drew the manual references.

### PI-CAI (Prostate Imaging: Cancer AI)

- **Name and maintainer:** PI-CAI, Radboud UMC (DIAG) and consortium. Source: https://pi-cai.grand-challenge.org/DATA/
- **Tasks and labels:**
  - Detection and diagnosis of clinically significant prostate cancer (csPCa, ISUP ≥ 2).
  - Voxel-level lesion masks (values 2–5 for csPCa lesions, empty otherwise).
  - Patient-level outcomes.
  - AI-derived lesion masks and whole-gland masks are also released.
- **Label format:** Label masks in the `DIAGNijmegen/picai_labels` GitHub repo (human-expert original and resampled sets, AI-derived sets). Clinical variables include PSA, prostate volume, PSA density, age and scanner. The exact mask file extension is **not stated on the page read.**
- **Size:**
  - Public training set: 1,500 cases from 1,476 patients (Zenodo v2.0). Source: https://zenodo.org/records/6624726
  - 425 cases are csPCa-positive, but only 220 of those carry human-expert annotations. The page also describes hidden tuning (100), hidden testing (1,000) and a private training set (7,607).
  - Patient IDs: some patients have more than one case (1,500 cases, 1,476 patients).
- **License and access:**
  - "CC BY-NC 4.0" per the challenge data page and the Zenodo record metadata (`cc-by-nc-4.0`). Sources: https://pi-cai.grand-challenge.org/DATA/ and https://zenodo.org/records/6624726
  - **Access class: open** (Zenodo).
- **Image format and dimensionality:** MHA (`_t2w.mha`, `_hbv.mha`, `_adc.mha`, and optional `_sag.mha` / `_cor.mha`), 3D biparametric MRI. Sequences are not co-registered in the training data.
- **Primary citation:**
  - Saha A et al., "Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study", Lancet Oncology 25, 2024, doi:10.1016/S1470-2045(24)00220-1.
  - Data DOI: 10.5281/zenodo.6624726.
- **Pitfalls:**
  - Most positive cases have no human mask, and treating an unannotated positive as empty is a label error.
  - ADC intensities are not standardized across centers.
  - The public set includes 328 ProstateX cases, so it overlaps with ProstateX.
  - Split by patient, because 24 patients have multiple cases.

## CT

### LIDC-IDRI

- **Name and maintainer:** Lung Image Database Consortium and Image Database Resource Initiative (NCI, FNIH, FDA), hosted by TCIA. Source: https://www.cancerimagingarchive.net/collection/lidc-idri/
- **Tasks and labels:**
  - Detection and segmentation: per-slice nodule outlines from up to 4 radiologists. Marks fall into three categories: nodule ≥3 mm, nodule <3 mm, and non-nodule.
  - Classification: subtlety, calcification, spiculation, lobulation, sphericity, margin, texture, and likelihood of malignancy on a 1–5 scale.
  - Patient-level pathology exists for only 268 scans (26.3%). Source: Armato 2011 (PMC3041807)
- **Label format:** XML, one file per scan, in `LIDC-XML-only.zip`. Marks are grouped by reader and not linked across readers, so users must cluster them by proximity. Source: TCIA page and Armato 2011.
- **Size:**
  - TCIA lists 1,010 subjects, 1,308 studies and 244,527 images (CT plus DX/CR). The paper gives 1,018 CT scans from 1,010 patients, with 8 patients scanned twice.
  - 7,371 lesions were marked as nodules by at least one reader; 2,669 nodules ≥3 mm were marked by at least one reader and 928 by all four.
  - Patient IDs: yes (per-subject collection IDs).
- **License and access:** "CC BY 3.0" per the TCIA collection page, which also calls it a "free publicly available" database. Source: https://www.cancerimagingarchive.net/collection/lidc-idri/
  - **Access class: open.**
- **Image format and dimensionality:** DICOM, 3D chest CT. The collection also contains 2D DX/CR radiographs.
- **Primary citation:**
  - Armato SG III et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans", Medical Physics 38(2), 2011, doi:10.1118/1.3528204.
  - Data DOI: 10.7937/K9/TCIA.2015.LO9QL9SX.
- **Pitfalls:**
  - No forced consensus, so inter-reader variability is large, and the unblinded second read makes reader marks correlated.
  - Reader identities and ordering vary between files.
  - Malignancy ratings are subjective; pathology is sparse.
  - Group the 8 patients with two scans by patient.

### LUNA16

- **Name and maintainer:** LUNA16 (LUng Nodule Analysis 2016), Radboud UMC (DIAG) and collaborators, hosted on grand-challenge.org. It is a subset of LIDC-IDRI. Source: https://luna16.grand-challenge.org/Data/
- **Tasks and labels:**
  - Detection (points with a diameter): the reference standard is nodules ≥3 mm accepted by at least 3 of 4 radiologists.
  - False-positive reduction (candidate classification, 0/1).
  - Excluded findings are listed in `annotations_excluded.csv`.
  - Automatic lung masks are provided but are not a reference standard.
- **Label format:** CSV.
  - `annotations.csv`: SeriesInstanceUID, world x/y/z and diameter in mm; 1,186 nodules.
  - `candidates_V2.csv`: candidate coordinates plus class; 1,166 of 1,186 nodules are detected.
- **Size:** 888 CT scans (LIDC scans with slice thickness >2.5 mm excluded), in 10 subsets meant for 10-fold cross-validation. Scans are keyed by SeriesInstanceUID. Patient IDs are not in the CSVs; map them through LIDC-IDRI DICOM metadata.
- **License and access:**
  - The page states "The data for LUNA16 is made available under … the Creative Commons Attribution 4.0 International License", while the underlying LIDC-IDRI is CC BY 3.0 Unported. Source: https://luna16.grand-challenge.org/Data/
  - Whether downloading needs a grand-challenge account: **not verified** (the Download page was not read).
  - **Access class: open** (per the license statement); registration requirements unverified.
- **Image format and dimensionality:** MetaImage (.mhd/.raw), 3D CT. Original DICOMs come from LIDC-IDRI.
- **Primary citation:** Setio AAA et al., "Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge", Medical Image Analysis 42, 2017, doi:10.1016/j.media.2017.06.015.
- **Pitfalls:**
  - Use the provided subsets for cross-validation. There is no held-out test set, so the dataset "is used for both training and testing".
  - Do not combine LUNA16 with LIDC-IDRI in a train/test split, because they are the same scans.
  - Whether the subsets keep a patient's two scans together is **not verified.**

### DeepLesion

- **Name and maintainer:** DeepLesion, NIH Clinical Center (Imaging Biomarkers and CAD Lab, R. Summers).
- **Tasks and labels:**
  - Detection with 2D boxes: universal lesion detection. Boxes come from RECIST diameters with 5-pixel padding.
  - Coarse body-region lesion type (8 classes) for 9,816 lesions only.
  - No segmentations and no 3D boxes. Source: Yan 2018 (PMC6052252)
- **Label format:** The paper does not name the annotation file. A `DL_info.csv` file is commonly cited, but this is **unverified** because the official Box folder could not be rendered.
- **Size:** 32,735 lesions in 32,120 key CT slices from 10,594 studies of 4,427 unique patients. The paper's split is random at the patient level, 70/15/15. Source: PMC6052252
- **License and access:**
  - **Not stated in the DeepLesion paper** (PMC6052252), which only says the data "may be downloaded".
  - The official host https://nihcc.app.box.com/v/DeepLesion did not render without JavaScript, so its terms are **not verified**.
  - Third-party mirrors give conflicting licenses (CC BY-SA 4.0 on Hugging Face; "unrestricted" on Academic Torrents). Neither is primary, so do not rely on either.
  - **Access class: open** (public Box link, no application described).
- **Image format and dimensionality:** 16-bit PNG slices converted from DICOM. Each key slice comes with neighboring slices, giving 2D or 2.5D context; not full 3D volumes.
- **Primary citation:** Yan K, Wang X, Lu L, Summers RM, "DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning", J Med Imaging 5(3):036501, 2018, doi:10.1117/1.JMI.5.3.036501.
- **Pitfalls:**
  - Labels are incomplete: radiologists bookmark only representative lesions, so many true lesions are unlabeled and count as false positives.
  - Some boxes measure normal structures.
  - Several studies per patient, so split by patient.

### KiTS23 (Kidney Tumor Segmentation Challenge 2023)

- **Name and maintainer:** KiTS23, N. Heller et al. (University of Minnesota / Cleveland Clinic team). Sources: https://kits-challenge.org/kits23/ and https://github.com/neheller/kits23
- **Tasks and labels:** Segmentation with integer labels 0 = background, 1 = kidney (parenchyma and hilar non-adipose tissue), 2 = tumor, 3 = cyst. Source: https://kits-challenge.org/kits23/
- **Label format:** NIfTI (`dataset/case_XXXXX/segmentation.nii.gz`; per-case `instances` folders hold the individual annotations). Source: GitHub API listing of `neheller/kits23/dataset`.
- **Size:** 599 cases: 489 training and 110 test. One case is one patient's most recent preoperative contrast-enhanced CT, in corticomedullary or nephrogenic phase. Case IDs are `case_00000…`.
- **License and access:**
  - "CC BY-NC-SA 4.0". The organizers do not count competing in the challenge as commercial use. Source: https://kits-challenge.org/kits23/
  - The repo README says "CC BY-NC-SA (Attribution-NonCommercial-ShareAlike)". The code is MIT.
  - Download: `kits23_download_data` pulls imaging from the Hugging Face dataset `neheller/KiTS-Challenge-Imaging`.
  - **Access class: open.**
- **Image format and dimensionality:** NIfTI 3D CT (`imaging.nii.gz` is downloaded separately from the masks).
- **Primary citation:**
  - Heller N et al., "The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT", arXiv:2307.01984 (2023). The repo names this as the citation; KiTS23 itself has no separate paper on the pages read.
  - Earlier: Heller N et al., "The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge", Medical Image Analysis 67, 2021, doi:10.1016/j.media.2020.101821.
- **Pitfalls:**
  - Contrast phase varies (KiTS19 was late-arterial only), and annotations come in multiple instances.
  - KiTS19, KiTS21 and KiTS23 overlap in cases, so do not mix editions across train and test.

### Medical Segmentation Decathlon (MSD)

- **Name and maintainer:** Medical Segmentation Decathlon (MICCAI 2018), A. Simpson, M. Antonelli, S. Bakas, M.J. Cardoso et al. Source: http://medicaldecathlon.com/
- **Tasks and labels:** Segmentation, 10 tasks:
  - CT: liver and tumor, lung tumor, pancreas and tumor, colon cancer, hepatic vessels and tumor, spleen.
  - MRI: brain tumor (BraTS 2016/17), hippocampus, prostate (central gland / peripheral zone), left atrium.
- **Label format:** NIfTI masks. All images were converted to NIfTI. Source: Antonelli 2022 (PMC9287542). A per-task `dataset.json` is **not verified** on a primary page.
- **Size:** 2,633 images in total. Train/test counts:
  - Brain 750 (484/266, 4D)
  - Heart 30 (20/10)
  - Hippocampus 394 (263/131)
  - Liver 201 on the site (paper table says 210; split 131/70)
  - Lung 96 (64/32)
  - Pancreas 420 (282/139)
  - Prostate 48 (32/16, 4D)
  - Colon 190 (126/64)
  - Hepatic vessels 443 (303/140)
  - Spleen 61 (41/20)

  Test labels are not released. Patient IDs: not stated; filenames were renamed.
- **License and access:** "CC-BY-SA 4.0". Sources: http://medicaldecathlon.com/ and Antonelli 2022. Downloads are offered via AWS and Google Drive.
  - **Access class: open.**
- **Image format and dimensionality:** NIfTI, 3D (4D for multi-sequence brain and prostate). MRI intensities were min-max scaled.
- **Primary citation:**
  - Antonelli M et al., "The Medical Segmentation Decathlon", Nature Communications 13, 2022, doi:10.1038/s41467-022-30695-9.
  - Simpson AL et al., "A large annotated medical image dataset for the development and evaluation of segmentation algorithms", arXiv:1902.09063 (2019).
- **Pitfalls:**
  - One rater per task, so inter-rater variability is unknown.
  - The paper says the hepatic-vessel labels are suboptimal.
  - Brain overlaps BraTS 2016/17 under renamed files, and liver comes from LiTS.
  - Several counts in the paper are internally inconsistent.

### TotalSegmentator (CT dataset; MR dataset noted)

- **Name and maintainer:** TotalSegmentator dataset, University Hospital Basel (Research and Analysis; J. Wasserthal et al.). Source: https://zenodo.org/records/10047292
- **Tasks and labels:** Segmentation of 117 anatomical structures in v2 (v1 had 104: 27 organs, 59 bones, 10 muscles, 8 vessels). Labels were made with model assistance and iterative refinement, and every case was manually reviewed (Wasserthal 2023).
- **Label format:** Per the TotalSegmentator README, the train/val/test split is in `meta.csv` inside the dataset. The mask file format is **not stated** on the Zenodo record, README or paper I read (NIfTI is likely but unverified).
- **Size:**
  - v2.0.1: 1,228 CT images. Source: https://zenodo.org/records/10047292
  - v1 paper: 1,204 exams, split 1,082 / 57 / 65 "patients" (PMC10546353). The paper does not say explicitly whether a patient can have more than one exam.
  - A separate MR dataset has 616 subjects. Source: https://github.com/wasserth/TotalSegmentator
- **License and access:** "Creative Commons Attribution 4.0 International (CC BY 4.0)" per the Zenodo record (license id `cc-by-4.0`). Source: https://zenodo.org/records/10047292
  - Model weights carry separate licenses (Apache-2.0 for the default task; some tasks are non-commercial only). Source: the GitHub README.
  - **Access class: open.**
- **Image format and dimensionality:** 3D CT. The file format is not stated on the pages read.
- **Primary citation:** Wasserthal J et al., "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images", Radiology: Artificial Intelligence 5, 2023, doi:10.1148/ryai.230024. Data: doi:10.5281/zenodo.10047292 (v2.0.1); v1 is doi:10.5281/zenodo.6802613.
- **Pitfalls:**
  - Mostly one institution and mostly Siemens scanners.
  - Labels are model-assisted, so a model trained on them may inherit nnU-Net biases.
  - Neighboring vertebrae and ribs are commonly confused.
  - v1 and v2 differ in classes and counts, so pin the version.

### RSNA Intracranial Hemorrhage Detection (2019)

- **Name and maintainer:** RSNA 2019 Brain CT Hemorrhage Challenge (RSNA with ASNR), hosted on Kaggle. Sources: https://www.rsna.org/rsnai/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019 and https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection
- **Tasks and labels:**
  - Classification at image (slice) level for five subtypes: subarachnoid, intraventricular, subdural, epidural and intraparenchymal hemorrhage.
  - Exam-level labels as well.
  - No boxes or ROIs; these were deliberately excluded. Source: Flanders 2020 (PMC8082297)
- **Label format:** The RSNA dataset description lists the annotation output as JSON from md.ai, with DICOM metadata/tags. The Kaggle label CSV layout is **not verified** (the Kaggle data page did not render). Source: the RSNA dataset-description PDF linked from the RSNA page.
- **Size:** 25,312 exams in the final dataset (21,784 train/val and 3,528 test) and 874,035 images, from 3 sites (Stanford, UNIFESP, Thomas Jefferson). The sets are "disjoint at the patient level", but the patient count is not stated.
- **License and access:**
  - "RSNA Intracranial Hemorrhage Detection Challenge Terms of Use and Attribution": use for "academic research and education, and other non-commercial purposes".
  - Terms: no re-identification, and redistribution needs a link plus a citation of Flanders 2020. Source: the terms PDF linked from https://www.rsna.org/rsnai/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019
  - The paper states it is "released under a noncommercial license".
  - Download is via Kaggle, which needs an account and acceptance of the competition rules (Kaggle rules **not verified**).
  - **Access class: registration.**
- **Image format and dimensionality:** DICOM axial slices, 512×512, mostly 5 mm thick. Labels are 2D per slice; slices can be stacked into 3D series.
- **Primary citation:** Flanders AE et al., "Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge", Radiology: Artificial Intelligence 2(3), 2020, doi:10.1148/ryai.2020190211.
- **Pitfalls:**
  - Most training exams were read by a single annotator (only the test set was triple-read).
  - Common errors: under-labeling, over-labeling and subtype confusion.
  - Site-specific case mix and anonymization signatures can leak site identity.
  - Split by series or patient, never by slice.
