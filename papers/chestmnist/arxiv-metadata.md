# arXiv metadata

For the human author to paste into the arXiv form. Nothing has been submitted.

- **Title:** Three Seeds of a ConvNeXt-Small Recipe on ChestMNIST, Scored with Recovered Patient Identifiers
- **Authors:** Isaiah R. Harville. arXiv's policy does not allow generative AI tools as authors, so Claude's part in the work is described in the Acknowledgment.
- **Affiliation:** Independent Researcher (confirmed by the author).
- **Comments:** 9 pages, 2 figures, 4 tables. Code, configurations, per-image predictions, and recovered patient identifiers at https://github.com/isaiah-harville/opendiagnostics
- **Primary category:** eess.IV (confirmed by the author)
- **Cross-list:** cs.CV (confirmed by the author)
- **License:** to be chosen by the author.

## Abstract

ChestMNIST holds 112,120 NIH chest radiographs with 14 findings mined from radiology reports. Its arrays carry no patient identifiers, so published results cannot check the split for patient overlap or resample patients for an interval. We recovered an NIH patient identifier for each ChestMNIST image by matching it to the NIH release. Among the 99.6% of images matched confidently, no patient is in two splits, and the test split is not NIH's own test list. We fine-tuned ConvNeXt-Small, pretrained on ImageNet-22k, on the 224-pixel images of the official split with one fixed recipe and three seeds. The mean test macro AUROC was 0.8450 (SD 0.0029). Each seed is above the best end-to-end fine-tuned value we found, 0.8322 (SD 0.0048), and so is each seed's 95% interval that resamples test patients. A Welch test gives p = 0.024. Each seed is also above the highest published value, 0.835, a 3-run mean with no SD from frozen foundation-model features, but the lowest seed's patient-level interval overlaps it, so that comparison is not resolved. We chose ChestMNIST after seeing its first result, and the recipe's interval clears the best published value on only 2 of 12 MedMNIST 2D subsets. Label accuracy at the 0.5 threshold is below that of every native 224-pixel baseline and of predicting every label negative, and we ran no ablation. We release the patient identifiers, every seed's per-image predictions, and the seed-0 weights.
