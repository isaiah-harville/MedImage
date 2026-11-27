# Claim

On the official TissueMNIST test split (47,280 images at 224 px), three training runs (seeds 0, 1, 2) of ConvNeXt-Small pretrained on ImageNet-22k and fine-tuned with one fixed MedMNIST recipe reached a mean macro one-vs-rest AUROC of 0.9637 (SD 0.0007) and a mean accuracy of 0.7823 (SD 0.0044); the 3-run means, and every run's 95% bootstrap interval over test images, lie above the best published TissueMNIST values under the same split and metric definitions (AUROC 0.9585 ± 0.0011 and accuracy 0.7592 ± 0.0038, Wu and Papież's end-to-end ViT-B/16 at native 224 px, mean of 3 runs; two-sided Welch test of 3 against 3 runs, p = 0.0042 for AUROC and 0.0026 for accuracy, or 0.0096 and 0.0034 if the published SDs divide by n).

The claim is about one recipe against published means. It does not say which part of the recipe causes the gap. S1 (two more seeds) is done: the 3-seed mean is 0.0052 AUROC above 0.9585, which is 5.7 pooled seed SDs, past the "about two pooled SDs" that S1 set as the bar for the recipe-level claim (`suggestions.md`). The earlier one-run claim is superseded.

## Contributions

1. TissueMNIST test results for three seeds, each with a bootstrap interval, released with per-image test and validation predictions so that later work can run a paired test against them.
2. A sourced, item-by-item comparison of our training setup with the MedMNIST+ setup (paper and code), which shows that the 224 px input images are the same as in the MedMNIST+ 224 px runs and lists the other differences as hypotheses with the run that would test each.
3. A per-class error analysis: in all three seeds, distal convoluted tubule (DCT) cells have the lowest AUROC and recall, and podocytes and glomerular endothelial cells are confused mainly with each other. (CD/CNT and PT, the largest classes, still hold 40% to 43% of the errors.)

## Evidence

Every number in the paper is a macro in `generated/numbers.tex`, written with the two generated tables by one command, `uv run python -I papers/tissuemnist/analysis/paper_numbers.py` (run from the repository root, about 6 min on CPU). It reads the three run folders and `models/tissuemnist-convnext/baselines.yaml`, recomputes every metric and bootstrap interval from the committed prediction files with the repository's metric functions, checks each against the run's `metrics.json` to 1e-9, checks AUROC and accuracy against the `medmnist` 3.0.2 evaluator (`getAUC`, `getACC`), checks that the three runs share test targets, data fingerprint, library versions, and configuration apart from the seed and later options left at their defaults, and exits with an error on any mismatch. Rerun on 2026-11-27: every check passed, and the values below match the numbers the training machine reported.

Runs: seed 0 `20261126-200655-ab39eb97` (trained at ab39eb9, rescored at e9f733f), seed 1 `20261127-113245-f5bc0be9` (f5bc0be), seed 2 `20261127-125653-a35c5dae` (a35c5da).

| Number in the paper | Value | Macro | Artifact |
|---|---|---|---|
| Test AUROC per seed, with interval | 0.9630 [0.9618, 0.9641], 0.9643 [0.9630, 0.9655], 0.9640 [0.9628, 0.9652] | `\testAurocSeed{Zero,One,Two}`, `Lo`, `Hi` | `predictions_test.csv`, `metrics.json` per run |
| Test accuracy per seed, with interval | 0.7772 [0.7739, 0.7811], 0.7853 [0.7820, 0.7889], 0.7844 [0.7809, 0.7880] | `\testAccSeed...` | same |
| 3-seed mean and SD, AUROC | 0.9637, 0.0007 | `\testAurocMean`, `\testAurocSD` | derived |
| 3-seed mean and SD, accuracy | 0.7823, 0.0044 | `\testAccMean`, `\testAccSD` | derived |
| Mean balanced accuracy, AUPRC | 0.7149, 0.7902 | `\testBalAccMean`, `\testAuprcMean` | derived |
| Lowest lower bound over seeds | AUROC 0.9618, accuracy 0.7739 | `\testAurocLoMin`, `\testAccLoMin` | derived |
| Mean bootstrap half-width | AUROC 0.0012, accuracy 0.0035 | `\testAurocHalfWidth`, `\testAccHalfWidth` | derived |
| Bootstrap SE (half-width / 1.96) | AUROC 0.0006, accuracy 0.0018 | `\testAurocBootSE`, `\testAccBootSE` | derived |
| Best published | 0.9585 ± 0.0011, 0.7592 ± 0.0038 | `\baseBestAuroc`, `\baseBestAurocSD`, `\baseBestAcc`, `\baseBestAccSD` | `baselines.yaml` (Wu and Papież, Table 6) |
| Mean minus best published | AUROC 0.0052, accuracy 0.0231 | `\gainMeanAuroc`, `\gainMeanAcc` | derived |
| Lowest lower bound minus best published | AUROC 0.0033, accuracy 0.0147 | `\marginLowerAuroc`, `\marginLowerAcc` | derived |
| Pooled SD, gap in pooled SDs | AUROC 0.0009, 5.7; accuracy 0.0041, 5.6 | `\pooledSD...`, `\gainPooledSD...` | derived |
| Welch t, df, two-sided p, and p if the published SD divides by n | AUROC 7.0, 3.3, 0.0042, 0.0096; accuracy 6.9, 3.9, 0.0026, 0.0034 | `\welchT...`, `\welchDf...`, `\welchP...`, `\welchPByN...` | `scipy.stats.ttest_ind_from_stats`; df by the Welch-Satterthwaite formula |
| Lowest seed minus best published, and in published SDs | AUROC 0.0045, 4; accuracy 0.0180, 5 | `\gainMinSeed...`, `\gainMinSeedInSD...` | derived |
| Error share of CD/CNT+PT and of DCT+GEC+podocytes, range over seeds | 40% to 43%; 24% to 26% | `\errShareCDCNTPTMin/Max`, `\errShareDCTGlomMin/Max` | per-run predictions |
| Test error rate | 22% | `\testErrorPct` | 1 minus mean accuracy |
| Subsets where the recipe's first-run interval, or point, is above the best published AUROC | 2 (ChestMNIST, TissueMNIST), 5, of 12 | `\subsetsAbove`, `\subsetsPointAbove`, `\subsetsTotal`; Table IV | `uv run python -I papers/tissuemnist/analysis/recipe_subsets.py` reads each `models/*-convnext` first run and `baselines.yaml` |
| MedMNIST+ DenseNet-121 at native 224, our mean gain | 0.9525, 0.0112 | `\baseNativeAuroc`, `\gainNativeAurocMean` | `baselines.yaml` |
| Validation means | AUROC 0.9630, accuracy 0.7859 | `\valAurocMean`, `\valAccMean` | `predictions_val.csv` |
| Per-class AUROC and recall, mean and range | DCT 0.9311, recall 0.458 (0.412 to 0.487) ... GEC 0.9777 | `\auroc<Class>Mean/Min/Max`, `\recall<Class>...` | per-run predictions |
| Confusions, mean over seeds | DCT as CD/CNT 30%, as TAL 11%; podocytes as GEC 22%; GEC as podocytes 16% | `\conf...` | same |
| Selected epoch | 7 of 8 in every seed | `\selectedEpochSeed...` | `metrics.json` `selection` |
| EMA minus saved raw val AUROC | 0.0021, 0.0003, 0.0009; mean 0.0011 | `\emaMinusRawAuroc...` | `metrics.csv`, `metrics.json` |
| Largest spread of val AUROC over last 3 epochs | 0.0006 | `\lastThreeSpreadMax` | `metrics.csv` |
| Run wall time | 99, 83, 83 min | `\runMinutesSeed...` | `run.json` |
| Parameters | 49,457,768 | `\parameters` | `run.json` |
| Table II, Table III | | `generated/tables/results.tex`, `perclass.tex` | written by the same command; baseline labels from `analysis/baseline_rows.tsv` |
| Fig. 1, Fig. 2 | | `figures/comparison.pdf`, `figures/training_curves.pdf` | `uv run python -I papers/tissuemnist/make_figures.py` |

Numbers in the text that are not run outputs, with their source:

| Number | Source |
|---|---|
| 236,386 cells, 165,466 training images, 3 specimens, 32x32x7 stacks | `yang2023medmnistv2` (ledger rows 13-15) |
| Published seed SDs 0.0002 to 0.0025 AUROC | the 3-run rows of Table II, from `baselines.yaml` |
| MedMNIST+ end-to-end AUROC 0.9220 to 0.9525 at 224 px; 0.9535 at 128 px | `doerrich2025medmnistplus` (rows 30-33) |
| Linear probe accuracy 0.6278 | `wu2025rethinking` (row 71) |
| ConvNeXt-Tiny 0.9519 AUROC at 64 px; ConvNeXt-T and -B 72.6% and 69.1% accuracy | `arman2026cytoecanet`, `manzari2023medvit` (row 63) |
| Woloshuk 58.3% cross-specimen balanced accuracy | `woloshuk2021kidney` (row 27) |
| ResNet-50 75.3% to 80.4%; SD 0.10 over 100 seeds | `wightman2021resnet` |
| about 29 runs per side | `bouthillier2021variance` |
| 2,586 steps per epoch, S = 20,688, EMA decay 0.99952, final lr 4.8e-6 | config: 165,466 / 64 rounded up; 8 epochs; 1 - 10/S; CosineAnnealingLR with T_max 8, eta_min 1e-6, at epoch 7 |
| 48-pixel border | (224 - 128) / 2 |
| sqrt(3/2) divisor factor | n = 3: sqrt(n / (n - 1)) |
| 0.449 and 0.226 gray mean and std | `config.yaml`; means of the ImageNet channel values |

## Vetting (explain the number)

- **Did the work happen?** All 47,280 test and 23,640 validation images have a prediction in every run, the per-class counts sum to 47,280, and the three runs' test targets are identical. Every run records the MD5 of the official `tissuemnist_224.npz` (`b077128c...`).
- **Same metric as the benchmark?** Yes. The `medmnist` evaluator gives the same AUROC and accuracy as our pipeline, to 1e-9, for every run.
- **Test split used once per run, after selection?** The recipe was committed at 6aa724b, before any TissueMNIST run. Seeds 1 and 2 changed only `seed_everything`. Selection used validation AUROC only, and all three runs selected epoch 7. No setting was chosen from test results.
- **Same protocol across seeds?** Seeds 1 and 2 trained 113 and 115 commits after seed 0. `git diff ab39eb9 a35c5da` on the training code shows only new options that default to off (`class_weight`, `variant`, k-fold splitting), reseeding of MONAI transforms (unused by this torchvision pipeline), and scikit-learn metrics (seed 0 was rescored with them at e9f733f). torch, timm, lightning, numpy, and scikit-learn versions are equal. `paper_numbers.py` asserts the configuration and versions.
- **What else could the number be measuring?** (a) Test-sampling noise: covered by each run's bootstrap. (b) Seed variation: measured, SD 0.0007 AUROC, 0.0044 accuracy; the gap is 5.7 and 5.6 pooled SDs. (c) Cell-level correlation: the bootstrap resamples images because there are no specimen IDs, so intervals are likely too narrow; the seed SD does not depend on this. (d) The EMA bug: test numbers use raw weights, which may understate the recipe. (e) A test-selected baseline: Wu and Papież report the best of three encoder learning rates without saying which split picked it, which would make the baseline optimistic, not ours. (f) An untuned baseline: the baselines are other groups' fixed recipes; the comparison is between training configurations, so it supports "this recipe beats those published results", not "ConvNeXt beats DenseNet or ViT".
- **Welch test caveat.** The sources do not say whether their SD divides by n or n-1. If n, the published SD is about 18% small and the test somewhat anti-conservative; the gap in pooled SDs (5.6 and above) leaves room for that.
- **Limiter (why not higher?).** DCT (recall 0.458) is mostly called CD/CNT or TAL, and podocytes and GEC are confused with each other. Each image is a 32 x 32 projection enlarged to 224 px.
- **Dataset selection.** The recipe ran once on all 12 MedMNIST 2D subsets at ab39eb9. Its interval clears the best published value on 2 (ChestMNIST, TissueMNIST); TissueMNIST was chosen for a paper after its first test result. The paper says so (Sec. VI-E) and does not adjust the p-values.
- **Verdict.** The 3-seed comparison claim holds. Which recipe difference causes the gap is open (S3, S4).

## Limits

- TissueMNIST was picked from 12 subsets after its first test result; the p-values do not adjust for that.
- Test-time flip averaging, which no baseline uses, is in every reported number (S3 would measure it).
- Three seeds per side; a rough estimate of run-to-run variation, and the Welch test has about 3 degrees of freedom.
- The interval resamples images, not specimens. All cells come from 3 specimens and the split is at the cell level, so the result says nothing about new donors, stains, or microscopes.
- No ablation. Network, pretraining, augmentation, regularization, schedule, selection rule, test-time flip averaging, and precision all differ from the baselines at once.
- The baselines' predictions are not public, so no paired test against them is possible.
- The saved weights are the raw weights; the EMA weights that guided selection were not saved.
- Seeds 1 and 2's checkpoints are recorded by SHA-256 but not uploaded (`artifacts.yaml` url is null); only seed 0's weights are public.
- GPU training ran with `deterministic: warn`, so a rerun may differ slightly.
- A research model on a benchmark, not a medical device.

## Authors

- Isaiah R. Harville, Independent Researcher (affiliation not yet confirmed by the author).
- Claude, Anthropic. Coauthor at the user's request (overrides the write-paper skill's no-AI-author rule; see decisions.tsv). IEEE's policy does not allow AI authors, and arXiv's policy must be checked before submission; main.tex carries a comment on this.
