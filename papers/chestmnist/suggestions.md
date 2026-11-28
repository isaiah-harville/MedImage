# Suggestions

What the paper needs before it can defend its claim, from the gap review after the literature review. Blocking items come first.

Cost basis: the published run took 53 min for 10 epochs on one RTX 4070 Ti, training plus per-epoch validation plus final scoring (`run.json`, 22:59:51 to 23:53:05 UTC), so about 5 min per epoch. The run has no `train.log`; `metrics.csv` has no wall-clock column.

## S1: Repeat the run with more seeds

- **Priority:** blocking for the original claim ("higher than every published ChestMNIST AUROC"); strengthens the narrowed claim in `claim.md`.
- **Run:** the unchanged recipe with seeds 1 to 4, on the same GPU and code commit:
  `uv run opendiagnostics fit -c models/chestmnist-convnext/config.yaml --seed_everything=<1..4>`.
  Then report the mean and SD of test macro AUROC over 5 seeds, and compare seed pairs with `uv run opendiagnostics compare`.
- **Why:** the claim rests on one run. The closest higher published values are `wu2025rethinking` (0.8322 +/- 0.0048 over 3 runs) and `hossain2026biofuse` (0.835, 3 runs). Wu and Papiez's SD for one end-to-end model, 0.0048, is about three quarters of our distance to BioFuse (0.0066) and half our distance to their own mean (0.0094). One run cannot separate differences that size.
- **Would change:** a 5-seed mean at or above about 0.840 with SD at or below 0.003 would support "higher than every published value we found". A mean near 0.835 would make the run a favourable draw, and the paper would say the recipe matches the best published values.
- **Without it:** the narrowed claim: one run reached 0.8416, 0.019 above the MedMNIST+ end-to-end best on the same images, and within the range of the best published values (0.832 to 0.835).
- **Cost:** 4 runs, about 3.5 GPU-hours on the RTX 4070 Ti. 2 runs (3 seeds in all, matching the baselines) cost about 1.8 hours.
- **Status:** done. Seeds 1 and 2 trained at a35c5da and 405f713 (runs 20261127-142008-a35c5dae and 20261127-150930-405f7138). Three seeds, not five: mean 0.8450, SD 0.0029, which meets the bar above on mean and SD. The claim against Wu and Papiez holds; against BioFuse it stays unresolved because seed 0's patient-level interval overlaps 0.835. Two more seeds would tighten the Welch tests.

## S2: Patient-level interval

- **Priority:** blocking for the original claim, because BioFuse's 0.835 sits 0.0019 below an image-level lower bound; strengthens the narrowed claim, which survives an interval up to about four times wider.
- **Run:** no training. Download NIH ChestX-ray14 (`Data_Entry_2017.csv` and the image archives, about 45 GB, open, no account). Downscale each NIH image to 224 px the way MedMNIST+ did and match it to the ChestMNIST test array by nearest pixel distance, to recover a patient ID per test row. Then rescore with `bootstrap_unit: patient`: `uv run opendiagnostics evaluate models/chestmnist-convnext/runs/20261126-165951-ab39eb97` after writing the IDs into `predictions_test.csv`'s `patient_id`, or call `bootstrap(..., groups=patient_ids)` from a script.
- **Why:** `metrics.json` records `"unit": "sample (no patient IDs)"`. ChestX-ray14 has 112,120 images of 30,805 patients (`yang2023medmnist`), so test images cluster by patient and the image bootstrap is too narrow by an unknown amount. Repo invariant 3 asks for a patient bootstrap.
- **Would change:** the interval width, and so whether 0.835 lies outside it.
- **Without it:** the paper says the interval resamples images and cannot rank the run against BioFuse.
- **Cost:** the download (hours, depending on bandwidth) plus about 1 to 3 CPU-hours of matching. Note the repo's AGENTS.md lets open datasets be downloaded without asking.
- **Status:** done. NIH IDs recovered by thumbnail matching (`scripts/patient_ids.py`, commit d1edb30); patient-level intervals with 10,000 replicates in `analysis/paper_numbers.py`. Widths are 1.2 to 1.3 times the image-level ones.

## S3: Train/test patient overlap

- **Priority:** strengthens
- **Run:** with the patient IDs from S2 for all three splits, count patients in both train and test, then rescore test AUROC on test patients absent from training.
- **Why:** MedMNIST v2 says it uses the official split; ChestX-ray14 v5 describes that split as patient-level (`wang2017chestxray8`), but nothing checks it for the MedMNIST arrays, and `check_disjoint` could not run. If patients overlap, a larger pretrained model could gain more from memorizing them than DenseNet-121 does.
- **Would change:** a drop in AUROC on unseen patients would weaken the comparison with every baseline.
- **Cost:** minutes once S2 is done.
- **Status:** done. 0 patients shared between train and test among confident matches; 49 if unsure matches count, and dropping their 607 test images moves AUROC by at most 0.0003. The split is not NIH's test_list.txt (5,190 of 22,433 test images are in it).

## S4: One-factor ablations of the recipe

- **Priority:** strengthens. The narrowed claim credits no single component, so no ablation is blocking.
- **Run:** change one setting at a time from `models/chestmnist-convnext/config.yaml`, 3 seeds each, compare with `opendiagnostics compare` against the S1 runs:
  1. pretraining: `model_name: convnext_small.fb_in1k`;
  2. network: `densenet121` (timm, ImageNet-1k) in our recipe;
  3. schedule and length: cosine over 100 epochs, early stopping after 5 epochs without lower validation loss;
  4. selection: checkpoint with the lowest validation loss instead of highest validation AUROC;
  5. augmentation: training transforms reduced to `Normalize`;
  6. regularization: weight decay 0.01 and `drop_path_rate: 0`.
- **Why:** the old paper's Table I lists these as candidate causes of the 0.019 gap; none was tested. `farooq2026mambavision` reports ConvNeXt-T (ImageNet-1k) at 0.802 on the same benchmark, which makes 1 and 2 the most informative.
- **Would change:** which differences the paper may name as causes.
- **Without it:** the setup comparison stays a list of hypotheses.
- **Cost:** items 1, 4, 5, 6 at about 53 min per run, 3 seeds each: about 10.5 GPU-hours. Item 2 about 2 to 3 hours. Item 3 runs until early stopping, up to 100 epochs, about 3 to 25 hours for 3 seeds. About 16 to 40 GPU-hours in all.
- **Status:** open, listed in the paper's Discussion and Limitations.

## S5: Add the missing published baselines

- **Priority:** strengthens
- **Run:** no compute. Add `wu2025rethinking` (DINOv2 ViT-B/14 end-to-end 0.8322, ACC 0.9487), `hossain2026biofuse` (0.835, ACC 0.949), `farooq2026mambavision` (MambaVision-B 0.823 +/- 0.002, no ACC), and `zheng2023cmixer` (0.822, ACC 0.963) to `models/chestmnist-convnext/baselines.yaml` with their table references.
- **Why:** `baselines.yaml` stops at 0.823, so the model card and the old paper overstate the margin. This file is outside `papers/`, so this run did not edit it.
- **Would change:** the results table and the "highest published value" wording.
- **Cost:** 15 minutes.
- **Status:** done. Added SDs for Wu and MedMNIST+, MambaVision-B and ConvNeXt-T (Farooq Tables V and VI), and C-Mixer (Table II) to baselines.yaml.

## S6: Reproduce the MedMNIST+ DenseNet-121 in this pipeline

- **Priority:** strengthens
- **Run:** a config with timm `densenet121` (ImageNet-1k), 3-channel input, no augmentation, AdamW lr 1e-4 with the library default weight decay, timm cosine with `t_initial=100`, early stopping on validation loss with patience 5, selection on validation loss, fp32, 3 seeds.
- **Why:** no baseline publishes per-image predictions, so the comparison can only use published means. A reproduced baseline allows the paired test that repo invariant 3 prefers.
- **Would change:** a paired difference with an interval, in place of a comparison with a published mean.
- **Cost:** about 3 to 6 GPU-hours, depending on when early stopping ends each run.
- **Status:** open. Both reviewers named it the most useful next run.

## S7: Score the EMA weights

- **Priority:** optional
- **Run:** retrain with `checkpoint.save_weights_only: false` (or fix the pipeline so weights-only checkpoints hold the EMA average), select between raw and EMA weights on validation only, then score test once.
- **Why:** the reported model is the raw weights; the EMA weights scored 0.8512 against 0.8470 on validation at the same epoch.
- **Cost:** one run, about 1 GPU-hour, plus the pipeline fix.
- **Status:** open, listed in Limitations.

## S8: One-to-one patient matching

- **Priority:** strengthens (from reviewer round 1)
- **Run:** on the training machine with the NIH images, extend `scripts/patient_ids.py` to assign ChestMNIST images to NIH images one-to-one (Hungarian or greedy unique by correlation), searching all label sets for the 438 rows without a confident match, and report how many ChestMNIST label sets differ from `Data_Entry_2017.csv`.
- **Why:** 194 NIH images are the best match of more than one ChestMNIST image, 4 of them for two confident matches, and one ChestMNIST label set has no NIH counterpart.
- **Would change:** the count of unsure matches and the 49 loosely shared patients; the patient-level intervals by a little.
- **Cost:** about the same CPU time as the first matching run, plus the 42 GB of NIH images already on that machine.
- **Status:** open
