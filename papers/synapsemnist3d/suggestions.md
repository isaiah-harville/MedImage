# Suggestions

What the paper needs before it can defend its claim, from the gap review after the literature review. Written after the five-seed runs of both networks came in; the stage-2 claim (one run beats the published values) was already dropped because TopoCAM reports 0.985 for a Kinetics-pretrained R3D-18. The claim this list tests is the one in `claim.md`: the two networks cannot be told apart over five seeds, the R3D-18 arm reproduces TopoCAM's value, and seed spread on this test set is as wide as the spread of published end-to-end results.

No item is blocking for that claim. Blocking items would be ones that the narrowed claim depends on; it names the networks rather than crediting pretraining, and it says the difference is unresolved rather than picking a winner.

## S1: R3D-18 from scratch

- **Priority:** strengthens
- **Run:** the `synapsemnist3d-r3d18` config with `pretrained: false`, seeds 0 to 4, on the RTX 4070 Ti:
  `for s in 0 1 2 3 4; do uv run opendiagnostics fit -c models/synapsemnist3d-r3d18/config.yaml --name synapsemnist3d-r3d18-scratch --model.init_args.network.init_args.pretrained=false --seed_everything=$s; done`
  (a new project `models/synapsemnist3d-r3d18-scratch/` would hold the published runs)
- **Why:** the two arms differ in stem, max pool, input channels, normalization and initialization at once (`torchvision029video`, `monai162resnet`). Kinetics pretraining cannot be credited or ruled out without a run that changes only the weights. TopoCAM's from-scratch rows are other architectures (`nuwagira2026topocam`, Table 2).
- **Would change:** if R3D-18 from scratch matches the pretrained R3D-18, any later gap is architecture; if it matches the MONAI ResNet-18, it is pretraining. Either result would add a sentence to Results and remove a limit.
- **Cost:** about 2.5 GPU hours (five runs of about 30 min).
- **Status:** open

## S2: more seeds per arm

- **Priority:** strengthens
- **Run:** seeds 5 to 19 of both projects:
  `for s in $(seq 5 19); do uv run opendiagnostics fit -c models/synapsemnist3d-resnet18/config.yaml --seed_everything=$s; uv run opendiagnostics fit -c models/synapsemnist3d-r3d18/config.yaml --seed_everything=$s; done`
- **Why:** with five seeds a side, the Welch 95% interval on the difference of mean AUROC runs from -0.005 to 0.020, so a gap of 0.02 is not ruled out. With the pooled seed SD of about 0.008, detecting a 0.007 gap at 80% power needs about 20 seeds per arm. Bouthillier et al. (`bouthillier2021variance`) recommend at least 29 runs per method for their probability-of-improvement test.
- **Would change:** a resolved difference in either direction would replace "unresolved" in the claim.
- **Cost:** about 15 GPU hours.
- **Status:** open

## S3: a fine-tuning learning rate for the Kinetics arm

- **Priority:** strengthens
- **Run:** the `synapsemnist3d-r3d18` config with `--optimizer.init_args.lr=1e-4`, seeds 0 to 4 (TopoCAM used Adam at 1e-4).
- **Why:** both arms use the from-scratch recipe (AdamW at 1e-3). A pretrained network is usually fine-tuned at a lower rate, so the R3D-18 arm may be under-tuned (benchmark-checklist question 2). The claim does not pick a winner, so this does not block it, but a tuned Kinetics arm could open a gap.
- **Would change:** a higher mean for the tuned arm would mean the unresolved difference understates what pretraining can give.
- **Cost:** about 2.5 GPU hours.
- **Status:** open

## S4: test the averaged weights

- **Priority:** strengthens
- **Run:** a code change so the checkpoint holds the EMA weights that validation scored (for example `checkpoint.save_weights_only: false`, or saving the averaged model in `after_fit`), then retrain both arms.
- **Why:** selection used the EMA weights and test used the raw weights of the selected epoch (`lightning266checkpoint`). Three from-scratch seeds selected epochs 47 to 60, when the learning rate was still high, and their raw weights give inhibitory volumes a mean score of 0.50 to 0.79, so accuracy at 0.5 falls to 0.767 to 0.855.
- **Would change:** accuracy at 0.5 and calibration of the from-scratch arm, and perhaps its AUROC spread. Review round 1 found that raw-weight validation AUROC ranks test AUROC across the ten runs (Spearman 0.75, p 0.013) while the EMA validation AUROC used for selection does not, and from-scratch seed 2 drops from 0.9764 (EMA) to 0.9467 (raw) on validation. The paper now says part of the from-scratch spread may come from this. This run would settle it. Review round 3 found the selected epoch wins by at most 0.0005 EMA validation AUROC and called S4 blocking for the spread half of the claim; the claim was narrowed instead (spreads reported only as upper bounds, no comparison with published gaps). It is the first run to do. Exact command, after confirming that a full checkpoint stores the EMA weights: `for s in 0 1 2 3 4; do uv run opendiagnostics fit -c models/synapsemnist3d-resnet18/config.yaml --checkpoint.save_weights_only=false --seed_everything=$s; uv run opendiagnostics fit -c models/synapsemnist3d-r3d18/config.yaml --checkpoint.save_weights_only=false --seed_everything=$s; done` (not run, not verified on this Mac).
- **Cost:** a small code change plus 5 GPU hours.
- **Status:** open

## S5: TopoCAM's other video networks

- **Priority:** strengthens (raised from optional after review round 1)
- **Run:** `KineticsVideoResNet` with `architecture: mc3_18` and `r2plus1d_18`, five seeds each.
- **Why:** TopoCAM reports 0.968 (MC3-18) and 0.975 (R(2+1)D-18), run count not stated. Five seeds would show whether their order holds, which the reviewer named as the experiment that would support a general claim about seed spread versus published gaps. The paper no longer makes that claim.
- **Cost:** about 5 GPU hours.
- **Status:** open

## S6: near-duplicate volumes across splits

- **Priority:** blocking at stage 2 (all volumes come from one rat)
- **Run:** `uv run python -I papers/synapsemnist3d/scripts/near_duplicates.py --out papers/synapsemnist3d/generated/near_duplicates.json`
- **Status:** done. The highest Pearson correlation between any test volume and any of the 1407 training and validation volumes, over all 48 flips and axis orders, is 0.25. No test volume is above 0.8.

## S7: seeds for the from-scratch network

- **Priority:** blocking at stage 2 (one run)
- **Status:** done. Seeds 1 to 4 of `synapsemnist3d-resnet18` are published.

## S8: the published network itself

- **Priority:** blocking at stage 2 (TopoCAM's R3D-18 matches the one run)
- **Status:** done. Five seeds of `synapsemnist3d-r3d18` are published; the paired comparisons are in `generated/seeds.json`.

## S9: rerun the published baselines

- **Priority:** optional
- **Status:** declined. TopoCAM, PlaneCycle and Park et al. release no per-volume predictions, so a paired test against them would need our own reimplementation of each, which would test our reimplementation, not their result. Our R3D-18 arm is the closest such rerun.

## S11: original data source citation

- **Priority:** optional
- **Run:** find and read the paper that released the rat EM volume behind SynapseMNIST3D (MedMNIST v2 names its source; TopoCAM cites Wei et al.), add a ledger row, and cite it in the Introduction.
- **Why:** reviewer round 1, m13. Neither the reviewer nor the citation audit found it by search.
- **Status:** open

## S10: release weights for every seed

- **Priority:** optional
- **Run:** `uv run opendiagnostics publish <run> models/<project>` for the nine runs whose `artifacts.yaml` has `url: null`. This uploads checkpoints to the public bucket, which needs the user's go-ahead.
- **Why:** only the seed-0 from-scratch checkpoint has a public URL. Predictions, configs and metrics are in git for all ten runs.
- **Status:** open
