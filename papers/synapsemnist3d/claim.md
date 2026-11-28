# Claim

On the official SynapseMNIST3D test split at 64³ (352 volumes from one rat), five seeds of a Kinetics-400-pretrained torchvision R3D-18 reached a mean test AUROC of 0.9871 (SD 0.0040), consistent with the 0.985 that TopoCAM reports for this network (one-sample p = 0.30, a failure to reject, not a demonstration of equality), and five seeds of a MONAI 3D ResNet-18 trained from scratch with the same recipe reached 0.9799 (SD 0.0101). The difference of 0.0072 is not resolved: an exact permutation test over the ten seed values gives p = 0.20, the paired volume bootstrap of the two five-seed means gives a 95% interval of -0.0009 to 0.0155, and the uncorrected per-seed paired tests split two for R3D-18, one for ResNet-18 and two unresolved. The five-seed ensembles score 0.9898 (ResNet-18) and 0.9887 (R3D-18), a paired difference of -0.0011 [-0.0085, 0.0056]. Every test number comes from the raw weights of an epoch selected on EMA weights (the selected epoch wins by at most 0.0005 EMA validation AUROC, and within the from-scratch arm raw validation AUROC ranks test AUROC with Spearman 0.90 while EMA validation gives 0.00), so the measured seed spreads (0.025 and 0.010) mix seed variation with the choice of raw snapshot and are reported only as upper bounds, not compared with published gaps. The two configurations differ in stem, input handling and compute (62.8 vs 106.3 GFLOPs) as well as initialization, and neither was tuned, so the paper does not credit pretraining.

Title: "Two 3D ResNet-18 Configurations Are Not Separable in Five Seeds Each on SynapseMNIST3D".

History: the stage-1 claim was that one from-scratch run (0.9845 [0.9702, 0.9951]) beat the best published value then known (0.88, Park et al.). Stage 2 found TopoCAM's 0.985 and PlaneCycle's 0.973 and stopped. The user then trained four more seeds of the from-scratch network and five seeds of the Kinetics R3D-18. This claim replaces the stage-1 claim.

## Contributions

1. Five-seed test results on SynapseMNIST3D at 64³ for two 3D ResNet-18 networks under one recipe, each seed with a 95% bootstrap interval over test volumes, released with per-volume predictions so later work can run a paired test against them.
2. Paired comparisons of the two networks over the five-seed means and seed-level tests, with a check of how far the tested raw weights depart from the EMA weights used for selection.
3. A near-duplicate check of every test volume against the training and validation volumes in all 48 orientations, and an account of how SynapseMNIST3D was chosen from six MedMNIST 3D subsets.

## Evidence

Artifacts: `models/synapsemnist3d-resnet18/runs/*/` (five runs, written `S0` to `S4`) and `models/synapsemnist3d-r3d18/runs/*/` (`K0` to `K4`). Every macro in `generated/numbers.tex` comes from one command, run from the repository root, which recomputes every metric and interval from `predictions_*.csv`, checks it against each `metrics.json` to 1e-9 and against the medmnist evaluator, checks that all ten runs share one split and one test label vector, that the configs differ only in the network and the seed, that no run had a dirty tree, and that all ran with one package set on one GPU:

`uv run python -I papers/synapsemnist3d/analysis/paper_numbers.py` (ran 2026-11-27 local time, 30 s on CPU, all checks passed, 300 macros)

Tables: `uv run python -I papers/synapsemnist3d/analysis/tables.py` (reads `generated/seeds.json`, the 3D projects, and `git show 39e65a0:models/*/baselines.yaml`). Figures: `uv run --with matplotlib python -I papers/synapsemnist3d/make_figures.py`. Near-duplicates: `uv run python -I papers/synapsemnist3d/scripts/near_duplicates.py --out papers/synapsemnist3d/generated/near_duplicates.json` (needs the 64³ arrays; ran in stage 2).

| Number in the paper | Value | Macro | Source |
|---|---|---|---|
| Per-seed test AUROC [95% CI], ResNet-18 scratch | 0.9845 [0.9702, 0.9951], 0.9897 [0.9810, 0.9965], 0.9651 [0.9452, 0.9803], 0.9862 [0.9756, 0.9947], 0.9740 [0.9582, 0.9877] | `\AurocScratch<Seed>`, `...Lo`, `...Hi` | S0-S4 `predictions_test.csv`, `metrics.json` |
| Per-seed test AUROC [95% CI], R3D-18 Kinetics | 0.9856 [0.9756, 0.9940], 0.9810 [0.9692, 0.9914], 0.9890 [0.9798, 0.9961], 0.9888 [0.9776, 0.9965], 0.9912 [0.9841, 0.9969] | `\AurocKinetics<Seed>`... | K0-K4 |
| Per-seed accuracy at 0.5 | scratch 0.963, 0.960, 0.767, 0.855, 0.855; Kinetics 0.946, 0.926, 0.952, 0.957, 0.957 | `\AccScratch<Seed>`, `\AccKinetics<Seed>` | same |
| Mean, SD, min, max AUROC per arm | scratch 0.9799, 0.0101, 0.9651, 0.9897; Kinetics 0.9871, 0.0040, 0.9810, 0.9912 | `\meanAurocScratch`, `\sdAurocScratch`, ... | same |
| Mean accuracy at 0.5 per arm | 0.8801, 0.9477 | `\meanAccScratch`, `\meanAccKinetics` | same |
| Mean AUROC of scratch seeds 1-4 (after the choice) | 0.9787 | `\laterMeanAurocScratch` | same |
| Seed-level difference, permutation p (252 relabelings), Welch p and 95% CI, Mann-Whitney p | 0.0072, 0.20, 0.20 [-0.005, 0.020], 0.31 | `\seedDiff`, `\permP`, `\welchP`, `\welchLo`, `\welchHi`, `\mannWhitneyP` | AUROCs above |
| Per-seed paired difference R3D-18 minus ResNet-18 [95% CI], p (2000 replicates) | +0.0011 [-0.0070, 0.0107] 0.839; -0.0087 [-0.0182, -0.0002] 0.045; +0.0239 [0.0097, 0.0408] <0.001; +0.0026 [-0.0092, 0.0147] 0.672; +0.0172 [0.0044, 0.0318] 0.003 | `\pairDiff<Seed>`, ... | `opendiagnostics.metrics.compare.compare`, the code behind `opendiagnostics compare` |
| Paired volume bootstrap of the five-seed means: difference [95% CI], p | 0.0072 [-0.0009, 0.0155], 0.079 | `\pooledDiff`, `\pooledLo`, `\pooledHi`, `\pooledP` | same predictions |
| Five-seed mean interval per arm | scratch [0.9686, 0.9895]; Kinetics [0.9780, 0.9943] | `\pooledLoScratch`, ... | same |
| Seeds above TopoCAM 0.985 / intervals above / below | scratch 2/0/1; Kinetics 4/0/0 | `\aboveTopoScratch`, ... | `baselines.yaml` |
| Seeds above PlaneCycle 0.973 / intervals above | scratch 4/2; Kinetics 5/4 | `\abovePlaneScratch`, ... | `baselines.yaml` |
| Selected epoch, val threshold, mean score of inhibitory test volumes | per run, see `generated/seeds.json` | `\Epoch...`, `\Thr...`, `\PInh...` | `metrics.json`, `predictions_test.csv` |
| Near-duplicates: max and median best correlation, count above 0.8 | 0.25, 0.10, 0 of 352 | `\dupMax`, `\dupMedian`, `\dupCount` | `generated/near_duplicates.json` |
| Subsets whose first-run interval was above the best published value at the choice | 1 of 6 (SynapseMNIST3D) | `\subsetsAboveThen`, `\subsetsTotal` | `generated/subsets.tex`, `tables.py` |
| Raw vs EMA validation AUROC at the selected epoch; Spearman rho with test (raw, EMA); Levene p | per run in `generated/seeds.json` (S2: 0.9764 EMA, 0.9467 raw); 0.75 (p 0.013), 0.10 (p 0.78); 0.25 | `\\ValEma<Run>`, `\\ValRaw<Run>`, `\\rhoRawTest`, `\\rhoEmaTest`, `\\leveneP` | `metrics.csv`, `metrics.json` val |
| Test accuracy at the val Youden threshold, scratch seeds 2-4 | 0.909, 0.952, 0.909 | `\\AccThrScratch<Seed>` | `metrics.json` test.accuracy |
| Holm-adjusted per-seed p; probability of improvement; ensemble AUROC; spread tests; GFLOPs; val class counts; TopoCAM Table 9 accuracies rebuilt at our prevalence | 1.000, 0.135, <0.001, 1.000, 0.012; 0.72; 0.9898, 0.9887; Levene median 0.25, mean 0.034, F 0.096; 62.8, 106.3; 129/48, 1/48 = 0.021; 0.929, 0.943, 0.917 | `\\pairHolm<Seed>`, `\\probImprovement`, `\\ensemble<Arm>`, `\\leveneP`, `\\leveneMeanP`, `\\fTestP`, `\\gflops<Arm>`, `\\nValExc`, `\\valStep`, `\\topoRebuilt*` | predictions, `torch.utils.flop_counter` on CPU, `sources.tsv` TopoCAM Table 9 row |
| Training settings typed in main.tex | AdamW lr 1e-3, wd 0.05, batch 16, 100 epochs, cosine to 1e-6, affine p 0.8 / 0.17 rad / 10% / 4 voxels / bilinear / border, flips p 0.5, EMA decay 0.9987 with buffers, Kinetics mean and SD, archive MD5 43bd14eb..., 1/95 = 0.011 | none | each run's `config.yaml`; torchvision `R3D_18_Weights`; `run.json` dataset fingerprint; MONAI and Lightning defaults |
| Ensemble paired difference; t-interval of each arm's mean; within-arm Spearman (raw, EMA); max selection margin; MTDL and CNMTDL counts; GFLOPs ratio; from-scratch mean interval half-width | -0.0011 [-0.0085, 0.0056]; scratch [0.9673, 0.9925], Kinetics [0.9822, 0.9920]; scratch 0.90, 0.00, Kinetics 0.70, -0.90; 0.0005; every run above MTDL 0.951, every interval except scratch seed 2, all above CNMTDL 0.932; 1.7; 0.012 | `\\ensembleDiff`, `\\tLo<Arm>`, `\\rhoRaw<Arm>`, `\\rhoEma<Arm>`, `\\maxSelectionMargin`, `\\ciAboveMtdl*`, `\\gflopsRatio`, `\\halfWidthScratch` | predictions, `metrics.csv`, `baselines.yaml` |
| Other macros used in the text | Welch df 5.2, SD ratio 2.6, one-sample p vs TopoCAM 0.30 (Kinetics), prevalence 0.730, reference volumes 1407, ACC above TopoCAM 4 and 2 | `\\welchDf`, `\\sdRatio`, `\\oneSampleTopoPKinetics`, `\\prevalence`, `\\dupRef`, `\\accAboveTopo*` | as above |
| Published values | TopoCAM R3D-18 0.985/0.929; PlaneCycle 3D ViT-L 0.973/0.930; Park 0.880/0.861; MTDL 0.951/0.931 (3 seeds); CNMTDL 0.932/0.855 (5 seeds); MedMNIST v2 ResNet-50+3D 0.851/0.795 | `\baseTopoAuroc`, ... | `baselines.yaml`, each row with its source |
| Split sizes, test classes | 1230/177/352; 257 excitatory, 95 inhibitory | `\nTrain`, ... | `splits.json`, `predictions_test.csv` |
| Parameters, minutes per run | 33.16 M and 31 min; 33.17 M and 28 min | `\paramsScratch`, `\minutesScratch`, ... | `run.json` |
| Versions | RTX 4070 Ti, torch 2.15.0, torchvision 0.29.2, Lightning 2.6.6, MONAI 1.6.2, medmnist 3.0.2, CUDA 13.2 | `\gpu`, ... | `run.json` |

Every value the brief listed (0.9845, 0.9897, 0.9651, 0.9862, 0.9740; 0.9856, 0.9810, 0.9890, 0.9888, 0.9912; seed-2 accuracy 0.767) matched the recomputation.

## Vetting (principle-explain-the-number, benchmark-checklist read for an eval)

- **Did every prediction happen?** Each run has 352 unique test IDs with one probability each, the same IDs and labels in all ten runs, and AUROC and accuracy equal to the medmnist evaluator's.
- **Do the configs differ only in the network?** Yes, after removing `seed_everything`, the name and notes, and two null-default keys that the seed-0 from-scratch config predates. The code between the five training commits changed only a multiclass loss weight, the Kinetics network class, serving and provenance.
- **What does the network swap change?** More than pretraining. MONAI's ResNet-18 has a 7×7×7 stride-1 stem and a stride-2 max pool on one channel. torchvision's R3D-18 has a 3×7×7 stem with stride (1,2,2), no pool, three input channels with Kinetics normalization, so its feature maps keep twice the depth resolution. Parameter counts are 33.16 M and 33.17 M. The claim names the networks and does not credit pretraining.
- **Was each side tuned?** Neither. Both use the from-scratch recipe, fixed before any test score. A pretrained network may do better at a lower learning rate (S3). Because the claim picks no winner, this weakens only the reach of "unresolved".
- **Did test data inform a choice?** Checkpoint and threshold come from validation. The subset was chosen after the seed-0 test result (one of six subsets above its best published value then), and all later runs came after that choice. The mean of the four later from-scratch seeds is 0.9787.
- **Is the gap larger than the noise?** No. The difference of means, 0.0072, is smaller than the from-scratch SD (0.0101) and than the mean interval half-width of one run (0.012 and 0.009).
- **What else could the numbers measure?** Overlap between splits: checked, none above 0.8 correlation. Raw versus EMA weights: test uses raw weights of the selected epoch, which explains the poor accuracy at 0.5 of from-scratch seeds 2 to 4 (inhibitory mean scores 0.79, 0.54, 0.50) while their AUROC stays high.

Verdict: the narrowed claim holds. No item in `suggestions.md` blocks it.

## Limits

- Five seeds per arm. The test cannot rule out a difference of up to 0.020 in mean AUROC.
- The two networks differ in more than pretraining. No R3D-18 from scratch was run (S1).
- One recipe for both arms, tuned for neither (S3).
- Test numbers come from raw weights; selection used EMA weights (S4).
- Published baselines give one value each, with no SD; PlaneCycle's is a mean of five runs. No paired test against them is possible.
- All 1,759 volumes come from one rat brain imaged with one microscope. MedMNIST has no animal or synapse-group IDs beyond this, so the bootstrap treats volumes as independent.
- The subset was chosen after its first test result.
- Runs used `deterministic: warn` with bf16 on CUDA, so reruns may not match bit for bit.
- Only the seed-0 from-scratch checkpoint is public (S10).

## Authors

- Isaiah R. Harville, Independent Researcher (affiliation not confirmed by the author). Claude is credited in the Acknowledgment, not as an author.
- arXiv category suggestion: eess.IV primary, cs.CV cross-list.

## Published values typed into main.tex

These come from the sources, not from our runs, and each has a ledger row in `sources.tsv`: TopoCAM MC3-18 0.968, R(2+1)D-18 0.975, M3T 0.958, TopoCAM 0.984, scratch models 0.499-0.588 (`nuwagira2026topocam`); PlaneCycle 0.971 (`yu2026planecycle`); ILPOResNet-50 0.940, C-Mixer 0.866, multi-task Swin 0.8575 ± 0.0089 (rows 16-18); Bouthillier's 29 runs and 0.75. Derived gaps in the text: 0.017 = 0.985 - 0.968, 0.012 = 0.985 - 0.973. Training-dynamics phrases ("about 15 epochs", "near 0.6 for about 20 epochs", "after about 35") are read off `figures/training_curves.pdf`.
