# Claim

## Verdict

Stage 1 stopped the one-seed paper as inconclusive (see the history in `decisions.tsv`). Suggestions S1 to S3 are now done: three seeds, NIH patient identifiers recovered for ChestMNIST, and patient-level intervals. The claim below holds against the best end-to-end published value. Against the highest published value, BioFuse (0.835, no SD), it does not resolve, and the paper says so.

## Claim

On the official ChestMNIST split at 224 pixels, each of three seeds of one fixed ConvNeXt-Small (ImageNet-22k) recipe has a test macro AUROC above the best end-to-end fine-tuned published value (Wu and Papież, DINOv2 ViT-B/14, 0.8322 +/- 0.0048), and so does each seed's 95% patient-level bootstrap interval (mean 0.8450, SD 0.0029; lowest patient-level lower bound 0.8350; Welch p = 0.024, 0.29 after a Bonferroni factor of 12 for choosing ChestMNIST from 12 subsets). Each seed is also above BioFuse's 0.835, but seed 0's patient-level interval reaches it, so that comparison is unresolved. Among the 99.6% of ChestMNIST images matched confidently to NIH, no patient appears in two splits, and the test split is not NIH's `test_list.txt`.

## Contributions

1. NIH patient identifiers for ChestMNIST, the split checked for patient overlap, and the finding that ChestMNIST's test split is not NIH's test list (17,243 of its 22,433 test images are in NIH's `train_val_list.txt`).
2. Three seeds with image- and patient-level intervals and released per-image predictions.
3. A sourced item-by-item setup comparison with MedMNIST+, and the observation that ACC at 0.5 mostly counts true negatives (ours is below every native-224 baseline and the all-negative predictor).

## Evidence

Every number in the paper is a macro in `generated/numbers.tex`, written by

    uv run python -I papers/chestmnist/analysis/paper_numbers.py

(about 30 minutes of CPU, mostly the 10,000-replicate patient bootstraps). The script recomputes each run's metrics from `models/chestmnist-convnext/runs/<run>/predictions_{test,val}.csv` and fails unless they equal `metrics.json` to 1e-9 and the medmnist evaluator's `getAUC`/`getACC`. It recomputes the patient summary from `generated/patient_ids.csv.gz` and fails unless it equals `generated/patient_ids.json`, including the seed-0-seeded 1,000-replicate patient intervals that file holds. It checks that the three runs share configuration (apart from the seed and defaulted options) and library versions. Published values come from `models/chestmnist-convnext/baselines.yaml`, each with a `sources.tsv` row.

| Number | Value | Macro |
|---|---|---|
| Test AUROC, seeds 0, 1, 2 | 0.8416, 0.8464, 0.8470 | `\testAurocSeedZero` etc. |
| Mean, SD | 0.8450, 0.0029 | `\testAurocMean`, `\testAurocSD` |
| Image-level 95% CI (1,000 replicates) | [0.8369, 0.8461], [0.8415, 0.8512], [0.8422, 0.8515] | `\testAurocLoSeed*` |
| Patient-level 95% CI (10,000 replicates) | [0.8350, 0.8469], [0.8403, 0.8516], [0.8408, 0.8522] | `\patientAurocLoSeed*` |
| Welch vs Wu, p (n-1; n) | 0.024; 0.044 | `\welchPWu`, `\welchPWuByN` |
| Difference to Wu, Welch 95% CI | 0.0030 to 0.0226 | `\diffWuLo`, `\diffWuHi` |
| One-sample vs 0.835, p | 0.028 (0.030 vs 0.8355) | `\oneSamplePBest` |
| Seeds 1 and 2 only, vs Wu / vs 0.835 | 0.034 / 0.014 | `\laterWelchPWu`, `\laterOneSamplePBest` |
| Test ACC mean; all-negative ACC | 0.9470; 0.9474 | `\testAccMean`, `\allNegAcc` |
| Confident matches (train, val, test) | 78,164, 11,162, 22,356 | `\confident*` |
| Patients shared train/test (confident; loose) | 0; 49 | `\sharedTrainTest`, `\looseSharedTrainTest` |
| Shared under a random split (5 shuffles) | 7,670 to 7,802 | `\nullShared*` |
| Test images in NIH test list | 5,190 | `\testFromNihTest` |
| Recipe clears best published on | 2 of 12 subsets | `\subsetsAbove` |

Figures: `uv run --with matplotlib python -I papers/chestmnist/make_figures.py` (reads `generated/seeds.json` and the run files).

## Limits

- Three seeds; Welch tests with about 3 degrees of freedom; published SDs from three runs with unstated divisors; BioFuse reports no SD.
- ChestMNIST was chosen after its first result; p-values are given raw and with a factor of 12.
- Patient IDs come from thumbnail matching; the matching is not one-to-one (194 NIH images claimed twice, 4 by two confident matches), and 438 images have no confident match (S8).
- No ablation and no in-pipeline baseline, so no component is credited (S4, S6).
- Raw weights are scored; EMA weights chose the epoch (S7).
- Report-mined labels, one hospital, reduced resolution. ACC at 0.5 is below every native-224 baseline. A research model, not a medical device.
- BioFuse's RAD-DINO and CheXagent were trained on chest radiographs, RAD-DINO's on data including NIH's.

## Authors

- Isaiah R. Harville, Independent Researcher (affiliation confirmed).
- Claude is not an author; the Acknowledgment says how Claude was used.
- Suggested arXiv category: eess.IV primary, cs.CV cross-list.
