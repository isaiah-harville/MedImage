# arXiv metadata

For the human authors to paste into the arXiv form. Nothing has been submitted.

- **Title:** Three Seeds of a ConvNeXt-Small Recipe Score Above the Best Published TissueMNIST Results
- **Authors:** Isaiah R. Harville. arXiv's policy does not allow generative AI tools as authors, so Claude's part in the work is described in the Acknowledgment.
- **Affiliation:** Independent Researcher (confirmed by the author).
- **Comments:** 8 pages, 2 figures, 4 tables. Code, configurations, and per-image predictions at https://github.com/isaiah-harville/opendiagnostics
- **Primary category:** eess.IV (confirmed by the author)
- **Cross-list:** cs.CV (confirmed by the author)
- **License:** to be chosen by the authors.

## Abstract

TissueMNIST is an eight-class set of 236,386 kidney-cortex cell images. The best published values we found are 3-run means from an end-to-end ViT-B/16 at 224 pixels, with AUROC 0.9585 and accuracy 0.7592. We fine-tuned ConvNeXt-Small, pretrained on ImageNet-22k, on the 224-pixel images of the official split with one fixed recipe, and trained it with three seeds. The mean test AUROC was 0.9637 (SD 0.0007) and the mean accuracy 0.7823 (SD 0.0044). The lowest of the three seeds is 0.0045 AUROC above the published mean. A two-sided Welch test of our three runs against the published mean and SD gives p=0.0042 for AUROC and p=0.0026 for accuracy. These numbers average each test image with its horizontal mirror, which no baseline reports. Scored without the mirror, the means are 0.9631 AUROC and 0.7805 accuracy, still above the published mean (p=0.0056 and p=0.0029). A sourced, item-by-item comparison with the MedMNIST+ setup shows that our inputs are the same 224-pixel images that MedMNIST+ and Wu and Papież used, so input size does not explain the gap. Pretraining, augmentation, regularization, and schedule all differ at once, and we trained no ablation, so we name each as a hypothesis with the run that would test it. Distal convoluted tubule cells have the lowest recall, and the two glomerular classes are confused mainly with each other. Against our lists of published results, the same recipe clears the best value on 2 of 12 MedMNIST 2D subsets, and we chose TissueMNIST for this paper after seeing its first test result. We release the configuration, every seed's per-image predictions, and the seed-0 weights.
