# Outline

Claim (from `claim.md`): over three training seeds, the ConvNeXt-Small recipe's mean TissueMNIST test AUROC and accuracy lie above the best published 3-run means (Wu and Papież ViT-B/16), and every seed's 95% interval over test images lies above them too.

## Abstract

TissueMNIST and its difficulty; published best is a 3-run mean of 0.9585 / 0.7592; we fine-tune ConvNeXt-S (IN-22k) with one fixed recipe, 3 seeds; mean AUROC and accuracy with SD; every seed's interval above the published means; Welch test against the published mean and SD; the setup comparison shows inputs equal the MedMNIST+ 224 px inputs, other differences untested; DCT and the glomerular pair carry most errors.

## I. Introduction

- TissueMNIST: largest 2D MedMNIST set, hardest by accuracy; published methods stop near 0.76 accuracy. Sources: `yang2023medmnistv2`, `doerrich2025medmnistplus`, `wu2025rethinking`.
- Gap: benchmarks fix one recipe per network; the published TissueMNIST numbers are 3-run means without intervals or predictions.
- Claim and three contributions (results with seeds and released predictions; sourced setup comparison; per-class error analysis).

## II. Related work

- Benchmarks with fixed recipes: MedMNIST v2, MedMNIST+, Wu and Papież (foundation models, end to end and linear probe).
- TissueMNIST-specific or new architectures: CytoECA-Net, MedViT/MedViTV2, MedKAN, Complex Mixer, MedLiT, BSDA, others below our interval (ledger rows 62-91).
- Source data: BBBC051 and Woloshuk et al. (3D and 2D, own split, cross-specimen drop).
- Recipe components: ConvNeXt recipe, BiT on pretraining scale, test-time augmentation on MedMNIST (`medeiros2026tta`).

## III. Data

- BBBC051 to TissueMNIST; split sizes; class counts (`\count...`); no specimen IDs.
- How 224 px images are made (notes); our inputs equal MedMNIST+ 224 px inputs; MD5.

## IV. Method

- Table I (setup comparison, hand-written, sourced P/C/N/S).
- Network, training, EMA and its save bug, recipe generator.

## V. Evaluation protocol

- Metric definitions (medmnist evaluator), bootstrap over images, seeds, Welch test from summary statistics, selection on validation only, rescoring of seed 0.

## VI. Results

- A. Comparison with published results: Table II (generated `results.tex`), Fig. 1 (`comparison.pdf`): question "where do the three seeds sit against published means?". Numbers: per-seed values, mean/SD, gain, margins, Welch t and p, pooled-SD ratio.
- B. Seed variation versus test-sampling variation: seed SD against bootstrap half-width.
- C. Per-class results: Table III (generated `perclass.tex`), confusions.
- D. Training dynamics: Fig. 2 (`training_curves.pdf`), EMA minus raw gap, plateau.

## VII. Discussion

Which setup differences plausibly matter, each with the run that would test it (S2-S6).

## VIII. Limitations

From `claim.md` Limits.

## IX. Reproducibility and licensing

Commits, commands, environment, checkpoint URLs, scripts.

## X. Conclusion

Claim again; next runs.

## Figures and tables

| Item | Question | Kind | Source |
|---|---|---|---|
| Table I | How does our setup differ from MedMNIST+? | table, hand-written | `doerrich2025medmnistplus`, `doerrich2025medmnistpluscode`, run `config.yaml` |
| Table II | How do the three seeds compare with every published TissueMNIST result? | generated table | `paper_numbers.py` from runs and `baselines.yaml` |
| Fig. 1 | Where do the seeds' intervals sit against the published means and SDs? | plot | `make_figures.py` from `metrics.json` and `baselines.yaml` |
| Table III | Which classes does the model confuse? | generated table | `paper_numbers.py` |
| Fig. 2 | Did training converge, and how far are the saved raw weights from the EMA weights? | plot | `make_figures.py` from `metrics.csv`, `metrics.json` |
