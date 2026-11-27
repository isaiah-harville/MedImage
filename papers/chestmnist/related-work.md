# Related work

The closest work reports ChestMNIST test macro AUROC on the official MedMNIST split with the official evaluator. All values are read in full; see `sources.tsv`.

| Key | What it does | How this paper differs | Compared in the experiments? |
|---|---|---|---|
| `doerrich2025medmnistplus` | MedMNIST+: 10 networks, 4 resolutions, end-to-end and linear-probe training, 3 seeds, on native 28 to 224 px images. ChestMNIST best: end-to-end DenseNet-121 at 224, 0.8222 +/- 0.0014. | Same images, split, and metric. Our recipe changes the network, pretraining data, augmentation, schedule, selection metric, regularization, precision, and seeds (1 vs 3). | Yes, as published numbers. Not rerun. Its code (`doerrich2024code`) gives the setup for the line-by-line comparison. |
| `wu2025rethinking` | Foundation models and CNNs on MedMNIST at several sizes, end-to-end and linear probe, 3 runs. ChestMNIST best: DINOv2 ViT-B/14 end-to-end at 224, 0.8322 +/- 0.0048. | A supervised ImageNet-22k ConvNeXt with a different recipe. Their single-model seed SD is the best available estimate of run-to-run spread. | Not in `baselines.yaml`. Should be added as a published row (S5). |
| `hossain2026biofuse` | BioFuse: frozen embeddings from five foundation models, concatenated, XGBoost head, MedMNIST+ 224, 3 runs. ChestMNIST 0.835, ACC 0.949. | Fine-tunes one network end to end, no foundation-model fusion. BioFuse's RAD-DINO saw NIH chest X-rays in pretraining. | Not in `baselines.yaml`. Should be added (S5). |
| `farooq2026mambavision` | MambaVision variants and CNN/ViT baselines on ChestMNIST 224, 10 seeds. Best 0.823 +/- 0.002; ConvNeXt-T 0.802. | ConvNeXt-Small with ImageNet-22k weights at 0.8416 vs their ImageNet-1k ConvNeXt-T at 0.802 suggests the recipe or the weights matter; not tested. | Not in `baselines.yaml` (S5). |
| `manzari2025medvitv2` | MedViTV2, a KAN-integrated transformer; ChestMNIST 0.823, ACC 0.967 on 28 px images upscaled to 224; run count not stated. | Native 224 px inputs, not upscaled 28 px. | Yes, as a published number. |
| `manzari2023medvit` | MedViT; ChestMNIST 0.805 on upscaled 28 px. | As above. | Yes. |
| `zheng2023cmixer` | C-Mixer; ChestMNIST 0.822, ACC 0.963 at 28 px. Its baseline rows disagree with MedMNIST v2. | Native 224 px. | Not in `baselines.yaml`; old paper cited it in text. |
| `yang2023medmnist` | MedMNIST v2: the benchmark, split, evaluator, and ResNet/AutoML baselines (best 0.778). | Uses its split and evaluator at native 224 px. | Yes. |

No paper found reports a ChestMNIST AUROC at or above 0.837 under this protocol. Two higher values that are not comparable were checked: FedDRAW (0.8425, macro over 5 CheXpert-matched labels, CLAHE preprocessing) and MambaVision-S at 512 px (0.837, a 13-class NIH subset with a custom split).

Not read, so the search is not complete: the full text of "Performance Evaluation of CNN Architectures for Multi-label Chest X-ray Classification on ChestMNIST" (Springer 2026, paywalled, no abstract), the published Applied Intelligence version of C-Mixer, TCPA-MedMTL (abstract claims "superior AUC" on ChestMNIST with no number), and about a dozen paywalled MedViTV2 citers with no abstract in OpenAlex.
