# Suggestions

What the paper needs before it can defend its claim, from the gap review after the literature review. Blocking items come first.

Costs use the measured run: 8 epochs took 99 minutes on one RTX 4070 Ti (`run.json` started and finished), about 12.4 minutes per epoch of 2,586 steps including validation.

## S1: Two more seeds of the same configuration

- **Priority:** blocking
- **Run:** the committed config unchanged except the seed, from a clean tree, on the same RTX 4070 Ti:
  `uv run opendiagnostics fit -c models/tissuemnist-convnext/config.yaml --seed_everything=1` and the same with `--seed_everything=2`; then `uv run opendiagnostics publish runs/tissuemnist-convnext/<run-id> models/tissuemnist-convnext` for each. Report the 3-seed mean and SD of test AUROC and accuracy next to each run's bootstrap interval, and add both to `paper_numbers.py`. Keep the EMA bug as is for these runs, so the three seeds share one protocol (see S6).
- **Why:** every published comparison row is a mean of 3 runs (`wu2025rethinking` Table 6, `doerrich2025medmnistplus` Table S12, `arman2026cytoecanet` Table 5), and ours is one draw. The literature review moved the best published AUROC from 0.9535 to 0.9585 (`wu2025rethinking`, ViT-B/16), so our point lead is 0.0045 AUROC, not 0.0095, and our lower bound clears it by 0.0033. Published seed SDs on this dataset run from 0.0004 to 0.0025 AUROC (`doerrich2025medmnistplus`, `arman2026cytoecanet`). With one seed we cannot tell a recipe that beats 0.9585 on average from a lucky draw. A reviewer would reject "exceeds the best published TissueMNIST results" on this evidence.
- **Would change:** a 3-seed mean AUROC above 0.9585 by more than about two pooled seed SDs supports the comparison claim as a recipe-level result. A mean at or below 0.9585 reduces the paper to "one run", and the accuracy comparison (lead 0.0180, lower-bound margin 0.0147) would carry it instead.
- **Without it:** "One training run of ConvNeXt-Small (ImageNet-22k) reached a TissueMNIST test AUROC of 0.9630 (95% interval over test images 0.9618 to 0.9641) and accuracy 0.7772 (0.7739 to 0.7811); both intervals lie above the published 3-run means (best 0.9585 and 0.7592, Wu and Papież). Variation across training runs was not measured, so the result does not show that the recipe beats those methods on average."
- **Cost:** 2 runs x 99 min = about 3.3 GPU-hours, plus 2 publishes.
- **Status:** done (2026-11-27). Seeds 1 and 2 trained on the RTX 4070 Ti and were published at 405f713 (`20261127-113245-f5bc0be9`, `20261127-125653-a35c5dae`). `paper_numbers.py` recomputes all three runs and checks them against `metrics.json`. Test AUROC 0.9630, 0.9643, 0.9640 (mean 0.9637, SD 0.0007); accuracy 0.7772, 0.7853, 0.7844 (mean 0.7823, SD 0.0044). The mean is 0.0052 AUROC above 0.9585, 5.7 pooled SDs, so the comparison claim now stands as a 3-seed recipe-level result (`claim.md`). Every seed's interval lies above both published means.

## S2: Paired comparison against a reproduced baseline

- **Priority:** strengthens
- **Run:** reproduce the strongest published baseline in this pipeline so `opendiagnostics compare` gives a paired test on the same 47,280 test images. Either (a) Wu and Papież's ViT-B/16 (`vit_base_patch16_224`), end to end at native 224 px, AdamW, encoder lr 1e-5, classifier lr 1e-3 decayed by 0.9 every 200 iterations, 15,000 iterations, 3 seeds; or (b) MedMNIST+'s DenseNet-121 (`densenet121.ra_in1k`) at 224 px, AdamW lr 1e-4, weight decay 0.01, cosine over 100 epochs, early stopping after 5 epochs without lower val loss, no augmentation, fp32, 3 seeds. Then `uv run opendiagnostics compare <baseline>/predictions_test.csv models/tissuemnist-convnext/runs/<run>/predictions_test.csv`.
- **Why:** no baseline publishes predictions, so the paper can only set intervals against published means (AGENTS.md invariant 3 allows this, but a paired test is stronger). Also checks that our pipeline reproduces the published numbers.
- **Would change:** a reproduced baseline well below its published value would point to a pipeline difference; a paired CI on the AUROC difference that excludes 0 replaces the interval-versus-mean argument.
- **Cost:** (a) 15,000 iterations at an unstated batch size; at batch 64 about 6 epochs, roughly 1.5 to 2 GPU-hours per seed, 5 to 6 h for 3. (b) early stopping typically ends in 15 to 30 epochs; at about 10 to 12 min per epoch, 3 to 6 h per seed, 9 to 18 h for 3.
- **Status:** open

## S3: Rescore without test-time flip averaging

- **Priority:** strengthens (both reviewers in stage 7 asked for it; the abstract's numbers include flip averaging, which no baseline uses)
- **Run:** on the training machine, where all three checkpoints are, for each seed copy the local run folder, turn flip averaging off in the copy, and rescore it. No training. In bash from the repository root:
  `for r in 20261126-200655-ab39eb97 20261127-113245-f5bc0be9 20261127-125653-a35c5dae; do mkdir -p runs/tissuemnist-convnext-notta && cp -r runs/tissuemnist-convnext/$r runs/tissuemnist-convnext-notta/$r && sed -i 's/^    tta: true$/    tta: false/' runs/tissuemnist-convnext-notta/$r/config.yaml && uv run opendiagnostics evaluate runs/tissuemnist-convnext-notta/$r; done`
  Then copy the three `metrics.json` and `predictions_*.csv` files somewhere the paper script can read them (they are not runs to publish).
- **Why:** the method section names flip averaging as one of the differences from every baseline; this is the only one that costs no training to measure. `medeiros2026tta` reports that mixed test-time augmentation hurt accuracy on three other MedMNIST sets (it did not test TissueMNIST).
- **Would change:** if the no-flip mean still clears 0.9585 and 0.7592, the claim no longer depends on flip averaging; a large drop credits part of the gap to it, which baselines could adopt for free.
- **Cost:** inference on 70,920 images per seed, about 10 minutes each on the GPU.
- **Status:** open

## S4: Ablations of pretraining, augmentation, regularization, and network

- **Priority:** strengthens
- **Run:** one arm per change, each with 3 seeds, everything else as in the committed config: (a) `model_name: convnext_small.fb_in1k` (ImageNet-1k only); (b) no `RandomResizedCrop` and no flips in `train_transforms`; (c) `label_smoothing: 0`, `drop_path_rate: 0`, `weight_decay: 0.01`; (d) `densenet121.ra_in1k` with our recipe. Add the arms to `scripts/medmnist_recipes.py` or as separate model projects.
- **Why:** contribution 2 lists these as hypotheses; the paper cannot credit any of them without a run that removes it. `kolesnikov2020bit` shows the pretraining effect on natural images, with the caveat that small models gain little.
- **Would change:** names which difference carries the gap, which turns the setup comparison from a list of hypotheses into a finding.
- **Cost:** 4 arms x 3 seeds x 99 min = about 20 GPU-hours (arm d may be shorter or longer depending on DenseNet's speed).
- **Status:** open

## S5: A specimen-level split

- **Priority:** strengthens
- **Run:** a data module for BBBC051 (`broad2021bbbc051`, CC BY 3.0, no account) that keeps the 3 donor specimens apart: leave one specimen out, 3 folds, same recipe and 2D maximum projection resized to 224.
- **Why:** the cell-level split puts cells from all 3 donors in train and test, so the result says nothing about a new donor; `woloshuk2021kidney` measured 58.3% balanced accuracy on an unseen specimen against 80.26% within specimens. A reviewer will ask why the paper did not test this when the source data allows it.
- **Would change:** a large drop belongs in the Limitations as a measured number instead of a caveat.
- **Cost:** about a day to write and test the data module, then 3 folds x about 70 to 100 min.
- **Status:** open

## S6: Save the EMA weights

- **Priority:** optional
- **Run:** fix the pipeline so the checkpoint holds the EMA weights that selection used (Lightning skips callback save hooks when `save_weights_only` is true), then retrain one seed.
- **Why:** logged EMA validation AUROC at the selected epoch is above the saved raw weights' by 0.0021, 0.0003, and 0.0009 in seeds 0, 1, 2 (mean 0.0011, `\emaMinusRawAurocMean`), so the reported test numbers likely understate the recipe. Not needed for the claim, which only gets stronger.
- **Cost:** the fix plus 99 min per seed. Do it after S1, so the three seeds share one protocol.
- **Status:** open

## S7: Bring `baselines.yaml` up to date

- **Priority:** strengthens (reproducibility)
- **Run:** add rows for `wu2025rethinking` (ViT-B/16 and EfficientNet-B4, native 224, App. C Table 6), `arman2026cytoecanet` (DenseNet-121 from scratch and CytoECA-Net, native 64, Table 5), and `yang2025medkan` (MedKAN-B); fix the MedMNIST v2 URLs from `/tables/3` (Table 2) to `/tables/4` (Table 3); mark Google AutoML Vision's input size as not stated. 
- **Why:** `paper_numbers.py` takes the best published values from this file; as it stands it reports 0.9535 as the best, which the literature review shows is wrong.
- **Cost:** minutes.
- **Status:** done (2026-11-27). Added `auroc_sd` and `accuracy_sd` where the source gives them, the Wu and Papież EfficientNet-B4 row and Arman et al.'s CytoECA-Net row, pointed the MedMNIST v2 URLs at `/tables/4`, and marked AutoML Vision's input as not stated. The Wu ViT-B/16, Arman DenseNet-121, and MedKAN-B rows were already in. `paper_numbers.py` takes the best row (Wu ViT-B/16) and the native-224 DenseNet-121 row from the file.

## S8: Upload the seed-1 and seed-2 checkpoints

- **Priority:** optional (reproducibility)
- **Run:** upload both checkpoints to the public weights bucket the way seed 0's was (see `docs/weights.md`), so `artifacts.yaml` gets a URL. This uploads to a public bucket, so it needs the user's go-ahead.
- **Why:** the paper releases every seed's predictions but only seed 0's weights.
- **Status:** open

