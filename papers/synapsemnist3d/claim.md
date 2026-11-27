# Claim

On the official SynapseMNIST3D test split at 64³ (352 volumes), one training run of a 3D ResNet-18 trained from scratch reached a test AUROC of 0.9845, with a 95% percentile bootstrap interval of 0.9702 to 0.9951 over test volumes, and the whole interval lies above the highest published AUROC we found for this subset, 0.88 at 64³ (Park et al.).

Status: does not hold as written. Stage 2 found that TopoCAM (Nuwagira et al., ML4H 2025) reports a Kinetics-pretrained 3D ResNet-18 at 64³ on the official split with AUC 0.985, and PlaneCycle (Yu et al. 2026) reports 0.973 (mean of 5 runs). Our 0.9845 equals the first and the second lies inside our interval. See related-work.md. The run stopped at stage 2.

## Contributions

1. A test AUROC of 0.9845 [0.9702, 0.9951] on SynapseMNIST3D at 64³ from a plain 3D ResNet-18 with standard augmentation, with released predictions, splits, config, and checkpoint, so others can run a paired test against it.
2. A side-by-side account of how this setup differs from the published 64³ and 28³ setups (input resolution, end-to-end 3D training, augmentation, schedule, weight averaging, test-time flip), with the run that would test each difference.

## Evidence

All artifacts are under `models/synapsemnist3d-resnet18/runs/20261127-051633-f5029e64/` (written `RUN/` below). One command regenerates every macro and fails if a recomputed value differs from `metrics.json` by more than 1e-9:

`uv run python papers/synapsemnist3d/scripts/make_numbers.py` (run from the repository root; ran 2026-11-27, all values matched).

It recomputes each metric from `predictions_*.csv` with the repository's `bootstrap_draws` (1000 replicates, seed 0, percentile), checks that the train, val, and test ID sets are disjoint and sized 1230, 177, 352, and reads the baselines from `baselines.yaml`.

| Number in the paper | Value | Macro | Artifact |
|---|---|---|---|
| Test AUROC and 95% CI | 0.9845 [0.9702, 0.9951] | `\testAuroc`, `\testAurocLo`, `\testAurocHi` | `RUN/predictions_test.csv`, `RUN/metrics.json` |
| Test accuracy at 0.5 and CI | 0.9631 [0.9403, 0.9830] | `\testAccHalf`, `\testAccHalfLo`, `\testAccHalfHi` | same |
| Test AUPRC and CI | 0.9942 [0.9880, 0.9983] | `\testAuprc`... | same |
| Test Brier and CI | 0.0327 [0.0175, 0.0502] | `\testBrier`... | same |
| Test ECE (15 bins) and CI | 0.0371 [0.0239, 0.0586] | `\testEce`... | same |
| Test sensitivity and specificity at the val Youden threshold | 0.9650 [0.9427, 0.9847], 0.9368 [0.8804, 0.9798] | `\testSens`..., `\testSpec`... | same |
| Val Youden threshold | 0.718 | `\threshold` | `RUN/metrics.json` |
| Val AUROC, raw weights with flip TTA | 0.9630 [0.9285, 0.9886] | `\valAuroc`... | `RUN/predictions_val.csv` |
| Val AUROC logged at the selected epoch (EMA weights, no TTA, torchmetrics) | 0.9619 | `\valAurocLogged` | `RUN/metrics.csv` |
| Selected epoch, epochs trained | 82, 100 | `\selectedEpoch`, `\epochs` | `RUN/metrics.json`, `RUN/metrics.csv` |
| Split sizes | 1230 / 177 / 352 | `\nTrain`, `\nVal`, `\nTest` | `RUN/splits.json` |
| Test class counts | 257 excitatory, 95 inhibitory | `\nTestExc`, `\nTestInh` | `RUN/predictions_test.csv` |
| Val class counts | 129 excitatory, 48 inhibitory | `\nValExc`, `\nValInh` | `RUN/predictions_val.csv` |
| Bootstrap replicates at or below 0.88 | 0 of 1000 | `\bootBelowPark` | recomputed draws |
| Margin over Park et al., point and lower bound | 0.104, 0.090 | `\marginPark`, `\marginParkLo` | `metrics.json`, `baselines.yaml` |
| Park et al. 64³ AUROC and ACC | 0.88, 0.861 | `\parkAuroc`, `\parkAcc` | `baselines.yaml` (source: arXiv 2602.21636v1, Table 1) |
| MedMNIST v2 ResNet-50+3D 28³ AUROC and ACC | 0.851, 0.795 | `\medmnistRfiftyAuroc`, `\medmnistRfiftyAcc` | `baselines.yaml` (source: Yang et al. 2023, Table 4) |
| MedMNIST v2 ResNet-18+3D 28³ AUROC and ACC | 0.82, 0.745 | `\medmnistReighteenAuroc`, `\medmnistReighteenAcc` | same |
| Wall time of the run (train and evaluate) | 28 min | `\trainMinutes` | `RUN/run.json` started and finished |
| Parameters | 33.2 M | `\params` | `RUN/run.json` |
| Commit, GPU, versions | f5029e64, RTX 4070 Ti, torch 2.15.0, Lightning 2.6.6, MONAI 1.6.2 | `\commit`, `\gpu`, ... | `RUN/run.json` |

## Vetting (principle-explain-the-number, benchmark-checklist)

The result is an evaluation number, not a speed number, so the checklist questions are read for an eval.

- **Did every prediction happen?** 352 unique test IDs and 177 val IDs, one probability each, matching the official split sizes. Recomputed AUROC equals the stored value.
- **Does the label orientation hold?** Target 1 is excitatory synapse, the column is `prob/excitatory synapse`, and prevalence is 0.730, which matches 257 of 352.
- **Did test data inform a choice?** No. The checkpoint was chosen on val AUROC (epoch 82) and the threshold on val (Youden). The test split was scored once by the pipeline after training.
- **Were the baselines tuned?** They are published values from their authors, each the authors' own tuned setup. We did not rerun them. Park et al. used frozen MedDINOv3 features at 64³. MedMNIST v2 used 28³ inputs. No published row is a 3D CNN trained end to end at 64³ on this subset (to be confirmed in stage 2).
- **Is the gap larger than the noise?** The interval half-width is about 0.013. The gap to 0.88 is 0.104, and 0 of 1000 replicates fall at or below 0.88. This is one seed. The seed-to-seed spread is unmeasured, but the gap is about eight times the interval half-width.
- **Why not higher?** Not measured. Errors at the val threshold are 6 inhibitory volumes called excitatory and 9 excitatory volumes called inhibitory (from sensitivity and specificity). Whether these are label noise or model errors is not known.
- **What else could the number measure?** Near-duplicate or spatially overlapping crops shared by train and test. MedMNIST v2 split this subset at random by synapse, and all synapses come from one rat's brain volume. This affects every method on this split equally, so it does not bias the comparison, but it may inflate all absolute numbers. Not checked; see `suggestions.md`.

Verdict: the comparison holds as a one-run result on the official split. Seed spread, ablations, and the overlap check are open.

## Limits

- One training run (seed 0). No seed spread.
- No ablation. The paper cannot say which part of the setup produced the gap.
- The baselines differ in resolution (28³ for MedMNIST v2) or in method family (frozen 2D foundation-model features for Park et al.). No published 3D CNN trained end to end at 64³ on this subset was found.
- Baseline values are means of several trials or of unstated trials, and none has an interval or released predictions, so no paired test is possible.
- MedMNIST has no patient or animal IDs, and all volumes come from one animal. The bootstrap resamples volumes, which treats volumes as independent.
- The released checkpoint holds the raw weights of epoch 82, not the EMA average that validation scored during training (Lightning 2.6.6 with `save_weights_only` skips the EMA save hook). All test numbers come from the raw weights with flip TTA.
- The run used `deterministic: warn` on CUDA with bf16 mixed precision, so a rerun on the same GPU may not match bit for bit.
- Research model on electron microscopy of one rat brain. Says nothing about other animals, microscopes, or resolutions.

## Authors

- Isaiah R. Harville, affiliation placeholder, email placeholder. Filled in by the human authors.
