# Claim

## Verdict of stage 1

The old paper's headline does not hold as stated. Its `paper.yaml` description says the run is "above the best published 0.823", and its Results say "all ten published values lie below the lower end of the interval". The literature review found two higher published ChestMNIST AUROCs on the same split and metric that the old paper missed:

- BioFuse, 0.835, mean of 3 runs, frozen foundation-model embeddings with XGBoost on native 224 px images (`hossain2026biofuse`, Table 3). That is 0.0019 below our lower bound of 0.8369, and our interval resamples images, not patients, so it is too narrow by an unknown amount.
- Wu and Papiez, DINOv2 ViT-B/14 fine-tuned end to end at 224 px, 0.8322 +/- 0.0048 over 3 runs (`wu2025rethinking`, Appendix C). Their run-to-run SD of 0.0048 for one end-to-end model is the best available estimate of how far a single run moves. Our single run is 0.0094 above their mean, about two of their SDs.

With one seed, the claim "higher than every published ChestMNIST AUROC" is inconclusive. The run stops at stage 1, narrowed to the claim below, which holds. Supporting the original claim needs suggestions S1 and S2 in `suggestions.md`.

## Narrowed claim

On native 224-pixel ChestMNIST, with the official MedMNIST split and evaluator, one training run of an ImageNet-22k-pretrained ConvNeXt-Small reached a test macro AUROC of 0.8416 (95% bootstrap interval over test images, 0.8369 to 0.8461), 0.019 above the best end-to-end result of the MedMNIST+ benchmark on the same images (DenseNet-121, 0.8222 +/- 0.0014 over 3 seeds), while its label accuracy at the 0.5 threshold, 0.9470, is below the 0.9474 of predicting every label negative.

Why the narrowed claim holds (inferred, not measured): the gap of 0.019 is 4 times our interval's lower half-width (0.0047), 14 MedMNIST+ seed SDs (0.0014), and 4 times the largest single-model seed SD we found for end-to-end fine-tuning on ChestMNIST (0.0048, `wu2025rethinking`). Patient clustering would have to widen our interval about four-fold before 0.8222 fell inside it.

## Contributions

1. A single public run, with weights, per-image test and validation probabilities, and the resolved configuration, that reaches 0.8416 test macro AUROC on native 224 px ChestMNIST, 0.019 above the MedMNIST+ end-to-end best on the same images.
2. A line-by-line comparison of our recipe with the MedMNIST+ end-to-end setup, taken from its paper and released code, with the one-factor experiment that would test each difference. No difference is tested here.
3. A measurement showing that ACC at the 0.5 threshold barely separates models on native 224 px ChestMNIST: our run (0.9470) and every MedMNIST+ model (0.9431 to 0.9484) sit within 0.005 of the 0.9474 of the all-negative predictor, while AUROC spans 0.5593 to 0.8222 in the same table. MedViT, MedViTV2, and C-Mixer report 0.959 to 0.967 on 28 px inputs; we could not explain those values from their papers and make no claim about them.

## Evidence

All run files are in `models/chestmnist-convnext/runs/20261126-165951-ab39eb97/` (below, `RUN/`). `vet.py` prints every computed value; it was rerun on 2026-11-27 and matched each value here.

| Number in the paper | Value | Macro | Artifact | Command that regenerates it |
|---|---|---|---|---|
| Test macro AUROC | 0.8416 | `\testAuroc` | `RUN/predictions_test.csv`, `RUN/metrics.json` | `uv run python papers/chestmnist/vet.py` (sklearn and `medmnist.evaluator.getAUC` agree to all digits) |
| Test AUROC 95% interval | 0.8369 to 0.8461 | `\testAurocLo`, `\testAurocHi` | `RUN/metrics.json` | `vet.py`, repo `bootstrap()` with 1,000 image replicates, seed 0 |
| Test ACC at 0.5 | 0.9470 | `\testAcc` | `RUN/predictions_test.csv` | `vet.py` (`getACC`) |
| Test ACC 95% interval | 0.9461 to 0.9479 | `\testAccLo`, `\testAccHi` | `RUN/metrics.json` | read from `metrics.json` |
| ACC of all-negative predictor | 0.9474 | `\allNegAcc` | `RUN/predictions_test.csv` | `vet.py` |
| Share of label decisions predicted positive at 0.5 | 2.24% | `\predPosRate` | `RUN/predictions_test.csv` | `vet.py` |
| Share of label decisions positive in the test labels | 5.26% | `\labelPosRate` | `RUN/predictions_test.csv` | `vet.py` |
| Test label prevalence range | 0.19% (hernia) to 17.55% (infiltration) | `\prevMin`, `\prevMax` | `RUN/predictions_test.csv` | `vet.py` |
| Test images | 22,433 | `\nTest` | `RUN/predictions_test.csv` | `vet.py` |
| Per-label test AUROC | 0.7093 (infiltration) to 0.9321 (hernia) | table | `RUN/metrics.json` | read from `metrics.json` |
| Validation AUROC, saved raw weights | 0.8470 | `\valAurocRaw` | `RUN/metrics.json` | read from `metrics.json` |
| Validation AUROC, EMA weights, selected epoch | 0.8512 | `\valAurocEma` | `RUN/metrics.csv`, epoch 6 (0-based) | read from `metrics.csv` |
| Selected epoch | 7 of 10 (1-based) | `\bestEpoch` | `RUN/metrics.json` `selection.best_epoch` = 6 | read |
| Training time, one RTX 4070 Ti | 53 min | `\trainMinutes` | `RUN/run.json` started and finished | read |
| Parameters | 49,462,382 | `\nParams` | `RUN/run.json` | read |
| MedMNIST+ DenseNet-121 end-to-end, 224 | 0.8222 +/- 0.0014, ACC 0.9484 | `\mpAuroc` | `doerrich2025medmnistplus`, Supp. Table S3 | ledger |
| Gap to MedMNIST+ best | 0.0194 | `\gapMp` | computed from the two rows above | `vet.py` (to add) |
| Wu and Papiez DINOv2 end-to-end | 0.8322 +/- 0.0048 | `\wuAuroc` | `wu2025rethinking`, Appendix C | ledger |
| BioFuse | 0.835, ACC 0.949 | `\biofuseAuroc` | `hossain2026biofuse`, Table 3 | ledger |
| MedViTV2-L, 28 px upscaled | 0.823, ACC 0.967 | `\medvitAuroc` | `manzari2025medvitv2`, Table 5 | ledger |

Checks run in stage 1, from `vet.py` unless noted:

- Did the work happen? Both prediction files are complete: 22,433 test and 11,219 validation rows, 14 labels, no missing values, unique ids.
- Is it the benchmark's metric? Our AUROC equals `medmnist.evaluator.getAUC` and our ACC equals `getACC` (threshold `> 0.5`) on the same predictions.
- Was the test split used for selection? The recipe was committed in `6aa724b` at 19:46 UTC on 2026-11-26, before the run started at 22:59 UTC (`run.json`), and selection used validation AUROC only. Only one ChestMNIST run is in the repository history.
- Which weights were scored? The checkpoint holds the raw weights of epoch 7, not the EMA weights the selection saw. Lightning 2.6.6 `dump_checkpoint` skips callback `on_save_checkpoint` hooks when `weights_only` is set, and the config sets `save_weights_only: true`. The rescored validation AUROC (0.8470) differs from the logged EMA value (0.8512) by 40 times the 1e-4 metric-implementation tolerance noted in `metrics/classification.py`.
- Can row order stand in for patients? No. Adjacent test rows have mean score correlation 0.0006, the same as shuffled rows (0.0008), so a block bootstrap over consecutive rows equals the image bootstrap. It was dropped.
- What limits the number? Not measured. The weakest labels are infiltration (0.709) and pneumonia (0.760); report-mined labels bound every model (the NIH labeler's per-label F1 ranges from 88.1 for pneumonia to 100 for hernia on 900 reports, `wang2017chestxray8` v5 Table 18).

## Limits

- One seed. Seed spread is the largest uncertainty the interval leaves out. Published single-model seed SDs on ChestMNIST range from 0.0014 (MedMNIST+ DenseNet-121) to 0.0048 (Wu and Papiez DINOv2).
- The interval resamples images. ChestMNIST has no patient IDs. MedMNIST v2 says it uses the official NIH split; ChestX-ray14's v5 text describes that split as patient-level. Neither claim was checked against image-level patient IDs here, so patient overlap across splits and clustering within the test split are both unverified.
- Labels are mined from radiology reports, not read from images.
- The checkpoint holds raw weights; EMA only chose the epoch.
- No ablation. The comparison with MedMNIST+ changes about ten settings at once, so it identifies candidate causes, not causes.
- ACC at 0.5 is below the all-negative predictor. No accuracy claim is made, and the threshold was not tuned, by protocol.
- One dataset, at reduced resolution. A research model, not a medical device.
- BioFuse uses RAD-DINO, which was pretrained on NIH chest X-rays among other data, so its 0.835 may include test-set exposure. This makes the comparison with it less clean in either direction.

## Authors

- Isaiah R. Harville, <affiliation placeholder>, <email placeholder>.
- Claude (Anthropic) is not an author, because arXiv does not allow generative AI tools as authors. The Acknowledgment says how Claude was used.
- arXiv category: placeholder. Suggested primary eess.IV, cross-list cs.CV.
