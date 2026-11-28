# Outline

Claim: see `claim.md`. Body near six IEEE pages; the claim is narrow and the results are one table, one figure and a handful of tests.

- **Abstract.** SynapseMNIST3D, a small 3D EM benchmark with 352 test volumes; published 64³ results are single values without spread; we trained five seeds each of a from-scratch MONAI ResNet-18 and a Kinetics R3D-18 under one recipe; means 0.9799 and 0.9871, difference unresolved (permutation p 0.20, paired bootstrap interval of the mean difference crossing zero); scratch seed range 0.025 wider than TopoCAM's three video CNNs; selection disclosed.
- **Introduction.** The benchmark and its published values (MedMNIST v2 at 28³; TopoCAM, PlaneCycle, Park at 64³). Our first run matched TopoCAM; what is left is whether the numbers are separable. Claim, three contributions.
- **Related work.** Published SynapseMNIST3D results grouped: MedMNIST v2 and 28³ follow-ups; 64³ fine-tuned networks (TopoCAM, PlaneCycle); frozen features (Park); topological pipelines (MTDL, CNMTDL). Seed variance in benchmarks (Bouthillier).
- **Data.** Source volume, split, 64³ archive, near-duplicate check (Section, numbers \dup*).
- **Method.** The two networks and how they differ (table: setup); the shared recipe; EMA and raw weights.
- **Evaluation protocol.** Metrics, bootstrap over volumes, paired compare, seed-level tests, how published values are compared.
- **Results.** (1) Per-seed table (tab:seeds) and figure (fig:seeds): the two networks. (2) Against published values (tab:published). (3) Seed spread against published gaps. (4) Accuracy at 0.5 and calibration of the raw weights. (5) Training curves (fig:curves). (6) How SynapseMNIST3D was chosen (tab:subsets).
- **Limitations.** From claim.md.
- **Reproducibility and licensing.** Commands, what is released, CC BY 4.0.
- **Conclusion.** Claim again; the runs in suggestions.md S1-S3.

Figures and tables:

| Id | Question | Kind | Source |
|---|---|---|---|
| fig:seeds | Do the two networks' seeds overlap, and where do the published values fall? | plot | generated/seeds.json via make_figures.py |
| fig:curves | How fast does each network reach its validation plateau, and which epoch was chosen? | plot | metrics.csv via seeds.json |
| tab:seeds | Per-seed AUROC, ACC, and paired difference | table | analysis/tables.py |
| tab:published | Which published values, at what size and run count | table | baselines.yaml via tables.py |
| tab:setup | How the two networks differ | hand-written table (no results) | torchvision and MONAI source, configs |
| tab:subsets | How the subset was chosen | table | analysis/tables.py |
