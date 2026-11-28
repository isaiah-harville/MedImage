# Outline

Claim (from `claim.md`): on the official ChestMNIST split at 224 pixels, three seeds of one fixed ConvNeXt-Small recipe give a mean test macro AUROC above every published ChestMNIST value we found under the benchmark protocol, and every seed's 95% patient-level interval lies above the best end-to-end fine-tuned value (Wu and Papież, 0.8322). Against the best published value overall (BioFuse, 0.835, a 3-run mean with no SD) the lowest patient-level lower bound meets it within its reported precision, so for BioFuse the paper reports the gap and the one-sample test and does not claim the interval excludes it.

## Abstract

ChestMNIST, its labels and the best published values; recipe and three seeds; mean and SD; patient IDs recovered from NIH and the split check; patient-level intervals; Welch against Wu and Papież, one-sample against BioFuse; accuracy at 0.5 below the all-negative predictor; ChestMNIST chosen after seeing its first result, recipe clears 2 of 12 subsets.

## I. Introduction

- ChestMNIST: 112,120 NIH radiographs, 14 report-mined labels, official split, metric. Sources: `yang2023medmnist`, `wang2017chestxray8`.
- Gap: benchmarks fix one recipe; published ChestMNIST values are 3-run means without intervals or predictions; no ChestMNIST result reports a patient-level interval, since the arrays carry no patient IDs.
- Claim and contributions: (1) three seeds with image- and patient-level intervals and released predictions; (2) NIH patient IDs recovered for ChestMNIST, the split checked patient-disjoint, and a patient-level interval; (3) sourced setup comparison with MedMNIST+ and the accuracy-at-0.5 observation.

## II. Related work

- Fixed-recipe benchmarks: MedMNIST v2, MedMNIST+, Wu and Papież, MambaVision benchmark.
- ChestMNIST-specific results: BioFuse (frozen FMs plus XGBoost; RAD-DINO saw NIH), MedViT, MedViTV2, C-Mixer, SECA-Net study, MedLiT, others lower; not comparable (FedDRAW, PNW).
- Source data: ChestX-ray14, labeler accuracy, split description.
- Comparing training runs: recipe effect, seed variance (`wightman2021resnet`, `bouthillier2021variance`, `hossain2026secanet`).

## III. Data

- ChestMNIST arrays, labels, prevalence, no patient IDs.
- Recovering patient IDs (Table II: matches, patients per split, overlap, NIH test-list membership).

## IV. Method

- Table I: setup comparison with MedMNIST+ (P/C/T sources).
- Network, training, EMA save behavior, recipe generator, seeds.

## V. Evaluation protocol

- Metrics (medmnist evaluator), image and patient bootstrap, Welch and one-sample tests, selection on validation only, commits and protocol check.

## VI. Results

- A. Comparison with published results: Table III (results, generated), Fig. 1 (comparison).
- B. Patient clustering and seed variation: width ratio, patient SE vs seed SD, unseen-patient sensitivity.
- C. Accuracy at the 0.5 threshold.
- D. Per-label results: Table IV (generated).
- E. Training dynamics: Fig. 2.
- F. How ChestMNIST was chosen: Table V (subsets, generated).

## VII. Discussion

Setup differences as hypotheses, each with its run (S4, S6, S7); BioFuse's NIH exposure.

## VIII. Limitations

From `claim.md`.

## IX. Reproducibility and licensing

## X. Conclusion

## Figures and tables

| Item | Question | Kind | Source |
|---|---|---|---|
| Table I | How does our setup differ from MedMNIST+? | table, hand-written | `doerrich2025medmnistplus`, `doerrich2024code`, run `config.yaml` |
| Table II | Is the ChestMNIST split patient-disjoint? | numbers in text | `analysis/paper_numbers.py` from `generated/patient_ids.csv.gz` |
| Table III | How do the three seeds compare with every published ChestMNIST result? | generated table | `analysis/paper_numbers.py`, `analysis/tables.py` |
| Fig. 1 | Where do the seeds' image and patient intervals sit against published means? | plot | `make_figures.py` from `generated/seeds.json`, `baselines.yaml` |
| Table IV | Which findings does the model rank worst? | generated table | `analysis/paper_numbers.py` |
| Fig. 2 | Did training converge, and how far are the raw weights from the EMA weights? | plot | `make_figures.py` from `metrics.csv`, `metrics.json` |
| Table V | How does the recipe do on the other subsets? | generated table | `analysis/paper_numbers.py` via `papers/tissuemnist/analysis/recipe_subsets.py` |
