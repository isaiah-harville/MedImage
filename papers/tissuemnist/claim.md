# Claim

On the official TissueMNIST test split (47,280 images at 224 px), one training run of ConvNeXt-Small pretrained on ImageNet-22k and fine-tuned with a fixed MedMNIST recipe reached a macro one-vs-rest AUROC of 0.9630 (95% bootstrap interval over test images, 0.9618 to 0.9641) and an accuracy of 0.7772 (0.7739 to 0.7811), both intervals above the best TissueMNIST values published under the same split and metric definitions (AUROC 0.9585 and accuracy 0.7592, Wu and Papież's end-to-end ViT-B/16 at native 224 px, mean of 3 runs).

The claim is about one run against published means. It does not say which part of the recipe causes the gap, and it does not say how much the run would vary across seeds. Stage 2 (literature review, 2026-11-27) replaced the old "best published" values, MedMNIST+ DenseNet-121 at 0.9535 and 0.7425, with Wu and Papież (`wu2025rethinking`, App. C Table 6); see `related-work.md`. Stage 3 marks this claim as needing more seeds before it can stand as a comparison (`suggestions.md` S1); the claim above is the narrower one-run version.

## Contributions

1. A TissueMNIST test result with a bootstrap interval, released with its per-image test and validation predictions so that later work can run a paired test against it.
2. A sourced, item-by-item comparison of this run's training setup with the MedMNIST+ setup (paper and code), which shows that the 224 px input images are the same as in the MedMNIST+ 224 px runs and lists the other differences as hypotheses with the run that would test each.
3. A per-class error analysis: distal convoluted tubule (DCT) cells and the two glomerular classes account for most errors.

## Evidence

All values below are printed by one command, `uv run python -I papers/tissuemnist/analysis/paper_numbers.py` (run from the repository root, 131 s on CPU), which reads `models/tissuemnist-convnext/runs/20261126-200655-ab39eb97/` and `models/tissuemnist-convnext/baselines.yaml`. It recomputes every metric and bootstrap interval from the committed prediction files with the repository's own metric functions, checks each against `metrics.json` to 1e-9, and checks AUROC and accuracy against the `medmnist` 3.0.2 evaluator (`getAUC`, `getACC`). It exits with an error on any mismatch. Rerun on 2026-11-27: every row below matched.

| Number in the paper | Value | Macro | Artifact | Command that regenerates it |
|---|---|---|---|---|
| Test macro AUROC | 0.9630 | `\testAuroc` | `runs/.../predictions_test.csv`, `metrics.json` | `paper_numbers.py` |
| Test AUROC 95% interval | 0.9618 to 0.9641 | `\testAurocLo`, `\testAurocHi` | same | same |
| Test accuracy (argmax) | 0.7772 [0.7739, 0.7811] | `\testAcc`, `\testAccLo`, `\testAccHi` | same | same |
| Test balanced accuracy | 0.7093 [0.7041, 0.7150] | `\testBalAcc...` | same | same |
| Test macro AUPRC | 0.7882 [0.7826, 0.7941] | `\testAuprc...` | same | same |
| Test images misclassified | 22% | `\testErrorPct` | same | same |
| Test images | 47,280 | `\testN` | same; matches `medmnist` INFO `n_samples.test` | same |
| Per-class test counts | CD/CNT 15,165 ... GEC 1,677 | `\count<Class>` | same | same |
| Per-class AUROC and interval | DCT 0.9297 [0.9243, 0.9351] ... GEC 0.9776 | `\auroc<Class>`, `Lo`, `Hi` | same | same |
| Per-class recall | DCT 0.412, Podo 0.589, GEC 0.722 ... | `\recall<Class>` | same | same |
| DCT predicted as CD/CNT, as TAL | 30%, 14% | `\confDCTasCDCNT`, `\confDCTasTAL` | same | same |
| Podocytes as GEC, GEC as podocytes | 24%, 14% | `\confPodoasGEC`, `\confGECasPodo` | same | same |
| Validation AUROC, accuracy (saved weights) | 0.9620 [0.9600, 0.9639], 0.7810 | `\valAuroc...`, `\valAcc...` | `predictions_val.csv` | same |
| Best published AUROC, accuracy | 0.9585, 0.7592 (script still prints 0.9535, 0.7425 until `baselines.yaml` gets the S7 rows) | `\baseBestAuroc`, `\baseBestAcc` | `baselines.yaml` | same |
| MedMNIST+ DenseNet-121 at native 224 | 0.9525, 0.7408 | `\baseNativeAuroc`, `\baseNativeAcc` | `baselines.yaml` | same |
| Our lower bound minus best published AUROC | 0.0033 (script prints 0.0083 against the stale best) | `\marginLowerAuroc` | derived | same |
| Point AUROC gain over best published | 0.0045 (script prints 0.0095 against the stale best) | `\gainPointAuroc` | derived | same |
| Our accuracy lower bound minus best published | 0.0147 (script prints 0.0314 against the stale best) | `\marginLowerAcc` | derived | same |
| Selected epoch, of 8 | 7 | `\selectedEpoch`, `\epochs` | `metrics.json` `selection`, `metrics.csv` | same |
| Logged (EMA) val AUROC, epoch 1, 7, 8 | 0.9253, 0.9641, 0.9638 | `\emaValAuroc...` | `metrics.csv` | same |
| EMA minus saved-weight val AUROC at epoch 7 | 0.0021 | `\emaMinusRawAuroc` | `metrics.csv`, `metrics.json` | same |
| Spread of logged val AUROC over last 3 epochs | 0.0006 | `\lastThreeSpread` | `metrics.csv` | same |
| Training loss, epoch 1 and 8 | 1.226, 0.871 | `\trainLossFirst`, `\trainLossLast` | `metrics.csv` | same |
| Run wall time | 99 min | `\runMinutes` | `run.json` started/finished | same |
| Parameters | 49,457,768 | `\parameters` | `run.json` | same |

## Vetting (explain the number)

- **Did the work happen?** All 47,280 test images and 23,640 validation images have a prediction, matching the `medmnist` 3.0.2 split sizes. The run's data fingerprint records the MD5 of the official `tissuemnist_224.npz` (`b077128c...`), which the data module checks on download. Per-class counts sum to 47,280.
- **Same metric as the benchmark?** Yes. The `medmnist` evaluator's `getAUC` and `getACC` give the same AUROC and accuracy as our pipeline, to 1e-9, on the committed predictions.
- **Test split used once, after selection?** The recipe was committed at 6aa724b, and the run trained at ab39eb9, the next commit, with no earlier TissueMNIST run in the history. Selection used validation AUROC only (`metrics.json` `selection`). The checkpoint was rescored at e9f733f (clean tree) after a change of metric library, with the same selection; the test numbers come from that rescoring.
- **What could the number be measuring instead?** (a) Test-sampling noise: covered by the bootstrap; the AUROC lower bound is 0.0033 above the best published mean (0.9585). (b) Seed-to-seed variation: not measured; one seed. Published seed SDs on TissueMNIST run from 0.0004 to 0.0025 AUROC (MedMNIST+ DenseNet-121 0.0007 and 0.0004, confirmed in Table S12; Wu and Papież ViT-B/16 0.0011; Arman et al. DenseNet-121 0.0025). Our point lead over the best published mean is 0.0045, between about 2 and 11 of those SDs, and ConvNeXt's own variation is unknown. (c) Cell-level correlation: the bootstrap resamples images because TissueMNIST has no specimen IDs; cells from 3 specimens are correlated, so the interval is likely too narrow. (d) The EMA bug: the saved checkpoint holds raw weights, not the EMA weights that guided selection; every test number is from the raw weights, so the bug can only have lowered our numbers. (e) An untuned baseline: the baselines are published values from a fixed setup applied to every MedMNIST dataset, and our recipe is also fixed across 12 datasets with no TissueMNIST search. The comparison is between two training configurations, so it supports "this run beats the published values", not "ConvNeXt beats DenseNet" or "our recipe beats theirs".
- **Limiter (why not higher?).** Errors concentrate in two places: DCT, recall 0.412, mostly called CD/CNT or TAL, and the GEC and podocyte pair, which the model separates well from the other classes but poorly from each other. Each image is a 32 x 32 maximum projection enlarged to 224 px, so the input holds little detail.
- **Verdict.** The claim, scoped to one run against published means, holds. Whether the recipe beats the baseline across seeds, and which difference causes the gap, are open; stage 3 decides whether they block.

## Limits

- One training seed. The bootstrap covers test sampling only, not training variation.
- The interval resamples images, not specimens. All cells come from 3 specimens and the split is at the cell level, so the result says nothing about new donors, stains, or microscopes.
- No ablation. Network, pretraining, augmentation, regularization, schedule, selection rule, test-time flip averaging, and precision all differ from the baseline at once.
- The baselines' predictions are not public, so no paired test against them is possible.
- The saved weights are the raw weights; the EMA weights that guided selection were not saved.
- GPU training ran with `deterministic: warn`, so a rerun may differ slightly.
- A research model on a benchmark, not a medical device.

## Authors

- Isaiah R. Harville, <affiliation placeholder>, <email placeholder>. Filled in by the human authors.
- Claude, Anthropic. Coauthor at the user's request (overrides the write-paper skill's no-AI-author rule; see decisions.tsv). IEEE and arXiv policies on AI authorship must be checked before submission.
