# Claim

On the official 224-pixel DermaMNIST test split (2,005 images), one end-to-end fine-tuning run of an ImageNet-22k ConvNeXt-Small, trained with augmentation, label smoothing, stochastic depth, and flip averaging at test time, reached a macro one-vs-rest AUROC of 0.9865 (95% image-level bootstrap interval 0.9809 to 0.9909), and that interval lies above the highest AUROC MedMNIST+ reports at the same resolution, 0.9683 (DINO ViT-B/16 linear probe, mean of three seeds), which used one shared setup with no augmentation.

The claim compares two training configurations. It does not say which of the differences between them produced the gap, and it does not say the model generalizes to new lesions.

## Contributions

1. A DermaMNIST result at 224 pixels with a bootstrap interval on every metric, the per-image test predictions, the configuration, and the checkpoint, so later work can run a paired test against it instead of comparing means.
2. A side-by-side of this training setup and the MedMNIST+ setup, taken from the MedMNIST+ paper and its public code, that lists each difference as a candidate explanation for the gap and names the run that would test it.
3. A per-class account of the errors that remain: melanoma against nevus, and actinic keratosis against benign keratosis.

## Evidence

All run artifacts are under `models/dermamnist-convnext/runs/20261126-141911-ab39eb97/` (called RUN below). `evidence.py` is `uv run python papers/dermamnist/analysis/evidence.py` from the repository root. It recomputes every metric from the per-image predictions with the pipeline's metric and bootstrap code (seed 0, 1,000 replicates), checks each value, lower bound, and upper bound against `RUN/metrics.json` to 1e-9, checks AUROC and accuracy against the official `medmnist.evaluator`, and prints one macro per line. Rerun on 2026-11-27: exit 0, every value below printed as listed.

| Number in the paper | Value | Macro | Artifact | Command that regenerates it |
|---|---|---|---|---|
| Test macro AUROC | 0.9865 | `\testAuroc` | RUN/predictions_test.csv, RUN/metrics.json | evidence.py |
| Test AUROC 95% interval | 0.9809, 0.9909 | `\testAurocLo`, `\testAurocHi` | same | evidence.py |
| Test accuracy (argmax) and interval | 0.9177 [0.9047, 0.9292] | `\testAcc`, `\testAccLo`, `\testAccHi` | same | evidence.py |
| Test balanced accuracy and interval | 0.8837 [0.8547, 0.9086] | `\testBacc`, `\testBaccLo`, `\testBaccHi` | same | evidence.py |
| Test macro AUPRC | 0.9249 | `\testAuprc` | same | evidence.py |
| Validation AUROC and interval, selected checkpoint | 0.9840 [0.9777, 0.9894] | `\valAuroc`, `\valAurocLo`, `\valAurocHi` | RUN/predictions_val.csv | evidence.py |
| Validation balanced accuracy and interval | 0.8259 [0.7688, 0.8812] | `\valBacc`, `\valBaccLo`, `\valBaccHi` | RUN/predictions_val.csv | evidence.py |
| Per-class test AUROC, interval, AUPRC, recall, count (7 classes) | see evidence.py output | `\aurocAkiec` ... `\recallVasc`, `\nAkiec` ... | RUN/predictions_test.csv | evidence.py |
| Test and validation sizes | 2,005 and 1,003 | `\nTest`, `\nVal` | RUN/predictions_*.csv | evidence.py |
| Nevi in test, accuracy of always answering nevus | 1,341, 0.669 | `\nNevusTest`, `\constantAcc` | RUN/predictions_test.csv | evidence.py |
| Melanomas called nevi, nevi called melanoma | 19, 44 | `\melAsNv`, `\nvAsMel` | RUN/predictions_test.csv | evidence.py |
| Actinic keratoses correct, called benign keratosis | 46 of 66, 12 | `\akiecHit`, `\akiecAsBkl` | RUN/predictions_test.csv | evidence.py |
| Selected epoch and its logged EMA validation AUROC | 27, 0.9847 | `\selectedEpoch`, `\selectedValAurocLogged` | RUN/metrics.json, RUN/metrics.csv | evidence.py |
| Validation AUROC at epoch 7, range over epochs 7 to 39 | 0.983, 0.981 to 0.985 | `\valAurocEpochSeven`, `\valAurocLateMin`, `\valAurocLateMax` | RUN/metrics.csv | evidence.py |
| Validation balanced accuracy at epoch 7 | 0.762 | `\valBaccEpochSeven` | RUN/metrics.csv | evidence.py |
| Wall time, fit and final evaluation, one RTX 4070 Ti | 24 min 33 s, 0.61 min per epoch | `\trainMinutes`, `\trainSeconds`, `\minutesPerEpoch` | RUN/run.json `started`, `finished` | evidence.py |
| Parameters | 49.5 M | `\parametersMillions` | RUN/run.json | evidence.py |
| Highest MedMNIST+ AUROC at 224 (DINO ViT-B/16, linear probe) | 0.9683 | `\bestPublishedAuroc` | models/dermamnist-convnext/baselines.yaml; MedMNIST+ arXiv 2404.15786v3, Supplementary Table S4 | evidence.py (reads baselines.yaml); Table S4 read on 2026-11-27, value 96.83 ± 0.07 |
| Highest MedMNIST+ accuracy at 224 (DenseNet-121, end to end) | 0.8474 | `\bestPublishedAcc` | same | same; Table S4 value 84.74 ± 0.51 |
| Our lower bound minus 0.9683; our point minus 0.9683 | 0.0126; 0.018 | `\gapToLower`, `\gapPoint` | RUN/metrics.json, baselines.yaml | evidence.py |
| Other baseline rows in the results table | baselines.yaml | none (table script) | models/dermamnist-convnext/baselines.yaml | table script, added at stage 5 |

## Vetting

Done with the principle-explain-the-number and benchmark-checklist skills. Benchmark-checklist questions about load, throughput, and hardware limits do not apply to an accuracy number; the questions below do.

- **Does it reproduce from the artifacts?** Yes. Every metric and interval recomputes exactly from the committed predictions (evidence.py, exit 0). The checkpoint was not re-run here: rescoring it needs the 198 MB checkpoint and the DermaMNIST 224 archive, and this machine has neither. So the link from checkpoint to predictions rests on the pipeline, which wrote both at a clean commit (`RUN/run.json` git.dirty false; `RUN/metrics.json` rescored.git.dirty false).
- **Is the metric the benchmark's?** Yes. AUROC and accuracy equal the official `medmnist.evaluator` `getAUC` and `getACC` on the same predictions to 1e-12. MedMNIST+ Table S4 labels its accuracy "for a fixed operating point of 0.5"; for a multiclass task the evaluator takes the argmax, as we do.
- **Did the work happen?** 2,005 test rows, every class present, one row per image ID. The test split was not used for selection: the checkpoint was chosen on validation AUROC (`RUN/metrics.json` selection), and the config was fixed in commit 6aa724b before the run (the run's commit ab39eb9 has the same config). This is the only DermaMNIST run in the repository.
- **Was each side tuned?** No side was tuned for DermaMNIST. Our recipe is shared across twelve MedMNIST subsets and differs per subset only in epochs and flips (`scripts/medmnist_recipes.py`). MedMNIST+ used one setup for all datasets by design, with no augmentation and early stopping. So the comparison is configuration against configuration, and the claim says so. It does not attribute the gap to the architecture, the pretraining, or any one component.
- **What else could the number measure?**
  - Seed luck. One run. MedMNIST+ reports seed standard deviations of 0.0006 to 0.0051 in AUROC for its three best 224-pixel DermaMNIST rows (Table S4). The gap from our lower bound to 0.9683 is 0.0126, about 2.5 times the largest of those. Not ruled out by a run of ours; see suggestions.md S1.
  - Lesion overlap between splits. HAM10000 has several images of some lesions and DermaMNIST carries no lesion ID, so test images may have siblings in training. That inflates every result on this split, and a model that fits training images more closely may gain more from it. Not ruled out; see suggestions.md S3. The claim is scoped to the official split for this reason.
  - Interval width. The interval resamples images, not lesions. Clustered images would widen it. Same cause as above.
- **What limits the number?** AUROC is near its ceiling for five of seven classes (bcc, df, vasc above 0.995). The macro mean is held down by melanoma (0.9678) and nevus (0.9809), and the largest error cells are melanoma against nevus. Balanced accuracy is held down by actinic keratoses (recall 0.697) and melanoma (0.825).
- **Verdict.** The number reproduces and is measured under the benchmark's protocol. The claim holds as a comparison of two configurations on the official split, with one run. It does not hold as a claim about any single component, about the architecture, or about new lesions.

## Stage 2 finding: a prior paper reports this result

The literature review (sources.tsv, related-work.md) found that Wu and Papiez (arXiv 2501.14685, Table 1) report DermaMNIST 224 end-to-end AUC 98.92 ± 0.11 for DINOv2 ViT-B/14, mean of three runs, on the same images and split. That value lies inside our interval and above our point estimate. Four more models in that table, and a ConvNeXt-Base and a ViT-Base in Sanap (doi:10.1007/s00521-026-12521-1, Table 6), reach 0.980 to 0.983. So the claim above is true as worded, since it names only MedMNIST+, but the comparison it implies is not: our run matches published end-to-end fine-tuning results and does not exceed them. The run stops at stage 2.

The narrower claim the evidence supports: on the official 224-pixel DermaMNIST split, one ConvNeXt-S fine-tuning run reaches AUROC 0.9865 [0.9809, 0.9909], no measurable difference from the best published end-to-end results (0.9892, mean of three runs), and on the 1,312 test images whose lesion is not in training its AUROC is 0.9795 [0.9674, 0.9883], against 0.9912 [0.9878, 0.9942] on the 693 test images whose lesion is.

### Added evidence

| Number | Value | Artifact | Command |
|---|---|---|---|
| Test images whose lesion also has a training image | 693 of 2,005 | Abhishek et al. `dermamnist_split_info.csv` and `CSV_files/combined_metadata_corrected.csv` (github.com/kakumarabhishek/Corrected-Skin-Image-Datasets, DermaMNIST/DermaMNIST_Analysis), RUN/predictions_test.csv | `uv run python papers/dermamnist/analysis/lesion_overlap.py <split_info.csv> <combined_metadata_corrected.csv>`; it first checks that all 2,005 mapped diagnoses equal the run's labels (they do) |
| AUROC, lesion also in training | 0.9912 [0.9878, 0.9942] | same | same |
| AUROC, lesion not in training | 0.9795 [0.9674, 0.9883], n = 1,312 | same | same |
| AUROC, lesion in neither training nor validation (equals the DermaMNIST-C test images in that metadata file) | 0.9795 [0.9674, 0.9892], n = 1,232; accuracy 0.9294, balanced accuracy 0.8687 | same | same |
| Best published end-to-end DermaMNIST 224 AUC | 98.92 ± 0.11 (DINOv2 ViT-B/14, 3 runs) | arXiv 2501.14685v1 Table 1 | read on 2026-11-27 |

The two subsets differ in class mix (melanoma is 20% of the overlap subset and 6% of the other), so the AUROC gap between them mixes leakage with composition. It is one run and an unpaired comparison of subsets. The DermaMNIST-C paper reports 1,227 test images; the metadata file used here gives 1,232, likely because the five test images in its newly found duplicate pairs are not moved in that file (not checked).

## Limits

- One training run, seed 0, with `deterministic: warn` on a GPU, so a rerun need not match bit for bit. The interval covers test sampling only, not seed variation.
- No ablations. Which differences from the MedMNIST+ setup produce the gap is unknown.
- The interval resamples images. DermaMNIST has no lesion or patient IDs, so leakage between splits could not be checked, and correlated images would widen the interval.
- The checkpoint holds the raw weights of epoch 27. EMA weights chose the epoch, but the pipeline saved raw weights (Lightning's `save_weights_only` skips the EMA callback's save hook), so all test numbers come from raw weights.
- No published DermaMNIST result has per-image predictions or intervals, so no paired test against a baseline is possible.
- One benchmark at one resolution. Nothing is claimed about 28, 64, or 128 pixels, other datasets, or clinical use.
- The data and weights carry the CC BY-NC 4.0 license of HAM10000 and DermaMNIST.

## Authors

- Isaiah R. Harville. Affiliation and email: placeholder, to be filled in by the author.
- Claude (Anthropic), kept as coauthor at the user's request, which overrides the write-paper skill's no-AI-author rule. The Acknowledgment says how Claude was used.
- arXiv category: unknown; suggested eess.IV with a cs.CV cross-list.
