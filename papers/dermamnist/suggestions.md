# Suggestions

What the paper needs before it can defend a claim. The run stopped at stage 2 because a prior paper already reports this result (see claim.md, "Stage 2 finding"), so these items are framed as the choices that decide what paper, if any, comes next. Blocking items come first.

Costs use the one measured run: 40 epochs of ConvNeXt-S at 224 pixels took 24 min 33 s including evaluation on one RTX 4070 Ti (RUN/run.json `started` and `finished`; there is no train.log in the run folder), about 0.61 min per epoch. RUN is `models/dermamnist-convnext/runs/20261126-141911-ab39eb97`.

## S1: Compare against the published end-to-end results that match ours

- **Priority:** blocking
- **Run:** no training. Add Wu and Papiez 2025 (arXiv 2501.14685, Table 1: DINOv2 ViT-B/14 end to end 98.92 ± 0.11 AUC, 92.02 ± 0.32 ACC; CLIP ViT-B/16 98.31; ViT-B/16 98.25) and Sanap 2026 (doi:10.1007/s00521-026-12521-1, Table 6: ConvNeXt-Base 0.9820, ViT-Base 0.9821, soft voting 0.9877) to `models/dermamnist-convnext/baselines.yaml` with their sources, and drop every sentence that says our run is above all published values.
- **Why:** `wu2025rethinking`, `sanap2026beyond`. Invariant 4 in AGENTS.md: every comparison uses the benchmark's published values. The old paper's headline claim is false without this.
- **Would change:** the headline becomes "no measurable difference from the best published end-to-end result", since 0.9892 lies inside our interval [0.9809, 0.9909].
- **Without it:** no claim against published values can be made. The paper could only report the run with its intervals.
- **Cost:** 15 minutes of editing; no compute.
- **Status:** open. Not done here: the brief limits writes to `papers/dermamnist/`.

## S2: Measure how much lesion overlap inflates the official-split score

- **Priority:** blocking, for the one claim no source in the ledger makes. Our run's AUROC is 0.9912 on the 693 test images whose lesion is in training and 0.9795 [0.9674, 0.9883] on the other 1,312 (`analysis/lesion_overlap.py`). That is one run, unpaired subsets, and a different class mix, so it shows the question is worth asking, not the answer.
- **Run:**
  1. Download DermaMNIST-C at 224 from Zenodo (doi:10.5281/zenodo.11101337, CC BY-NC 4.0, no account per the paper; about 1 GB, unchecked). Add it to the data module as a predefined split.
  2. Train the same config on the DermaMNIST-C training split, seeds 0, 1, 2: `uv run opendiagnostics fit -c models/dermamnist-convnext/config.yaml --seed_everything=<s>` with the data pointed at DermaMNIST-C.
  3. Train the official-split config with seeds 1 and 2 as well (S3 covers this).
  4. Score both sets of runs on the same lesion-disjoint test images (the 1,232 from `lesion_overlap.py`, or the 1,227 DermaMNIST-C test images) and on the 693 overlap images, and run `uv run opendiagnostics compare` between an official-split run and a DermaMNIST-C run on the shared test images.
  5. For a second architecture, repeat steps 2 and 3 with `timm` `vit_base_patch14_dinov2` end to end, so the inflation is measured on the model family with the best published score.
- **Why:** `abhishek2025investigating` shows the overlap exists (886 train-test images) and says the two benchmarks should not be compared, but reports no per-model inflation. The claim in claim.md is scoped to the official split for this reason.
- **Would change:** if official-split training scores clearly higher than DermaMNIST-C training on the same lesion-disjoint images, the official-split leaderboard, ours and Wu and Papiez's alike, partly measures recall of training lesions. That is a new, citable finding.
- **Without it:** the paper can only report the subset numbers above as one observation, in Limitations.
- **Cost:** ConvNeXt-S, 3 seeds on DermaMNIST-C: about 75 min (8,215 training images, about 17% more than 7,007, so about 29 min per run). DINOv2 ViT-B/14, 3 seeds per split: about 6 runs at an estimated 40 to 60 min each, 4 to 6 h (guess; patch 14 at 224 gives 256 tokens, not measured). Data prep and the data-module change: about 2 h of work. Total about 7 to 9 GPU hours.
- **Status:** open

## S3: Repeat the run over seeds

- **Priority:** strengthens
- **Run:** `uv run opendiagnostics fit -c models/dermamnist-convnext/config.yaml --seed_everything=<s>` for seeds 1 to 4, then report the mean and spread of test AUROC beside the bootstrap interval.
- **Why:** one run. The published comparisons are means of three runs. `baek2026detection` gives a ResNet-18 DermaMNIST AUC seed SD of 0.0037, and MedMNIST+ Table S4 gives 0.0006 to 0.0051 for its best rows.
- **Would change:** a seed spread near 0.004 would put Wu and Papiez's 0.9892 and our mean in the same range either way; it settles whether "no measurable difference" holds across seeds, not only across test resampling.
- **Without it:** every statement says "one run".
- **Cost:** 4 × 24.5 min ≈ 1 h 40 min.
- **Status:** open

## S4: Score the released checkpoint without flip averaging

- **Priority:** strengthens
- **Run:** copy RUN to a scratch folder, set `model.init_args.tta: false` in its `config.yaml`, download its checkpoint from the URL in `artifacts.yaml`, and run `uv run opendiagnostics evaluate <copy>`. Run it on a copy, because `evaluate` rewrites `metrics.json` and the predictions.
- **Why:** the old draft names flip averaging as a possible contributor; `medeiros2026believe` finds TTA hurts at 28 pixels but tested no flip-only TTA. This also rescores the checkpoint, which this session could not do (claim.md, Vetting).
- **Would change:** a drop of more than the interval's half-width would make TTA part of the recipe's result; a paired `compare` between the two prediction files gives the answer directly.
- **Without it:** TTA stays an untested part of the recipe.
- **Cost:** CPU inference on 3,008 images twice: about 10 to 20 min on this Mac (guess), plus downloading the 198 MB checkpoint and the DermaMNIST 224 archive.
- **Status:** open

## S5: Ablate augmentation and pretraining

- **Priority:** optional now that `wu2025rethinking` has already shown that the MedMNIST+ pipeline, not the architectures, explains the end-to-end gap.
- **Run:** three seeds each of (a) the config with `RandomResizedCrop` and the flips removed, and (b) the config with `model_name: convnext_small.fb_in1k`.
- **Why:** attribution of the gap to MedMNIST+ to any single component.
- **Would change:** which component carries the gap.
- **Without it:** no attribution, as now.
- **Cost:** 6 × 24.5 min ≈ 2 h 30 min.
- **Status:** open

## S6: Save the EMA weights the checkpoint was selected with

- **Priority:** optional
- **Run:** fix the pipeline so the checkpoint holds the averaged weights (`checkpoint.save_weights_only: true` in `src/opendiagnostics/cli.py` skips the EMA callback's save hook in Lightning 2.6.6), then retrain once.
- **Why:** the reported numbers come from raw weights chosen by EMA validation scores (claim.md, Limits).
- **Would change:** small, given the flat validation curve (0.981 to 0.985 from epoch 7 on).
- **Without it:** disclose it, as the draft does.
- **Cost:** one code change plus 24.5 min.
- **Status:** open

## S7: Paired test against the best published model

- **Priority:** optional
- **Run:** no published DermaMNIST result has per-image predictions. Either ask Wu and Papiez for theirs or train DINOv2 ViT-B/14 in this pipeline (part of S2, step 5) and run `uv run opendiagnostics compare`.
- **Why:** invariant 3: a claim that one model beats another needs a paired test.
- **Would change:** lets the paper say whether ConvNeXt-S and DINOv2 differ at all.
- **Without it:** only "inside our interval".
- **Cost:** covered by S2; asking the authors is a message outside this run.
- **Status:** open
