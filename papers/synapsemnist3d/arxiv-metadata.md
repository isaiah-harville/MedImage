# arXiv metadata

For the human authors to paste into the arXiv form. Nothing has been submitted.

- **Title:** Two 3D ResNet-18 Configurations Are Not Separable in Five Seeds Each on SynapseMNIST3D
- **Authors:** Isaiah R. Harville. arXiv's policy does not allow generative AI tools as authors, so Claude's part in the work is described in the Acknowledgment.
- **Affiliation:** Independent Researcher (not yet confirmed).
- **Comments:** 7 pages, 2 figures, 4 tables. Code, configurations, and per-volume predictions at https://github.com/isaiah-harville/opendiagnostics
- **Primary category:** eess.IV (suggested, not confirmed)
- **Cross-list:** cs.CV (suggested, not confirmed)
- **License:** to be chosen by the authors.

## Abstract

SynapseMNIST3D asks whether a 64^3 electron microscopy crop shows an excitatory or an inhibitory synapse. Its official test split holds only 352 volumes, all synapses of one neuron in one rat. None of the published results at 64^3 that we read reports a spread in its main text. The highest we found is 0.985 AUROC, for a Kinetics-400-pretrained R3D-18 in the TopoCAM paper, with no run count stated. We trained five seeds each of two 3D ResNet-18 configurations under one fixed, untuned recipe: a MONAI ResNet-18 from random weights and the torchvision R3D-18 from its Kinetics-400 weights. They differ in stem, input handling, and compute as well as in initialization. Mean test AUROC was 0.9799 (SD 0.0101) for the first and 0.9871 (SD 0.0040) for the second, consistent with TopoCAM's value. The difference, 0.0072, is not resolved. An exact permutation test over the ten seed values gives p = 0.20, and a paired bootstrap over test volumes puts the difference of the five-seed means between -0.0009 and 0.0155. The five-seed ensembles score 0.9898 and 0.9887. Every test number comes from the raw weights of an epoch selected on averaged weights, so the seed spreads we measured, 0.9651 to 0.9897 from scratch, mix seed variation with the choice of raw snapshot and are upper bounds. No test volume correlates above 0.25 with any training or validation volume. We chose this subset after its first result, and we release every seed's per-volume predictions.
