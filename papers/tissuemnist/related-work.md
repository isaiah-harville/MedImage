# Related work: the closest set

No paper found reports a TissueMNIST result at or above our interval (AUROC lower bound 0.9618, accuracy lower bound 0.7739) on the official split with full training data. 41 candidate papers were read in full or screened (see `decisions.tsv`, phase `literature`). The best published values are Wu and Papież's end-to-end ViT-B/16 on native 224 px images: accuracy 0.7592 ± 0.0038 and AUROC 0.9585 ± 0.0011, means of 3 runs. That is above the MedMNIST+ DenseNet-121 row (0.9535 / 0.7425) that the old draft called the best. `baselines.yaml` now holds the Wu and Papież, Arman et al., and MedKAN rows (S7).

All values are TissueMNIST test values, AUROC / accuracy. "Compared" says whether our experiments compare against the work: "published" means against its published number, with no paired test.

| Key | What it does | TissueMNIST result | How this paper differs | Compared |
|---|---|---|---|---|
| `wu2025rethinking` | Fine-tunes 12 pretrained CNNs and ViTs end to end and by linear probe on all MedMNIST 2D sets at 224 px, 15,000 iterations, encoder learning rate picked from 3 values, 3 runs. | ViT-B/16 0.9585 ± 0.0011 / 0.7592 ± 0.0038 (App. C Table 6); EfficientNet-B4 0.9569 / 0.7556; DenseNet-121 0.9522 / 0.7384. | Best published values. Its "best end-to-end" row is chosen over 3 learning rates with no stated validation rule, so it may be test-selected. One fixed generic recipe per network, no augmentation stated. Ours is one ConvNeXt-S run with a different recipe. | published |
| `doerrich2025medmnistplus` | Ten networks at 28/64/128/224 px, three training schemes, fixed setup (AdamW 1e-4, cosine over 100 epochs, early stopping on val loss, no augmentation), 3 seeds. | DenseNet-121 at 128 px zero-padded: 0.9535 ± 0.0007 / 0.7425 ± 0.0039; at 224 px: 0.9525 ± 0.0004 / 0.7408 ± 0.0032 (Supp. Table S12). | The benchmark whose setup the method section contrasts item by item. Our inputs equal its 224 px inputs. | published |
| `arman2026cytoecanet` | TissueMNIST-specific lightweight CNN with channel attention, native 64 px MedMNIST+ images, official split, 3 seeds. | DenseNet-121 from scratch 0.9572 ± 0.0025 / 0.7547 ± 0.0055; CytoECA-Net 0.9564 / 0.7556 ± 0.0006 (Table 5, Results text). | Smaller input and networks; reports seed SDs up to 0.0025 AUROC, which bears on our one-seed claim. | published |
| `yang2023medmnistv2` | Defines TissueMNIST and its split; ResNet-18/50 at 28 and 28-upscaled-to-224, three AutoML systems, at least 3 trials. | Best AutoKeras 0.941 / 0.703 (Table 3). | Benchmark definition and protocol source. | published |
| `yang2025medkan` | KAN-integrated network on MedMNIST, images resized to 224 from an unstated source size, run count not stated. | MedKAN-B 0.957 / 0.740 (arXiv 2502.18416v1, Table 1). | Different architecture; protocol details missing. | published |
| `manzari2023medvit`, `manzari2026medvitv2` | Hybrid CNN-transformer networks on 28 px images upscaled to 224, run count not stated. | MedViT-S 0.952 / 0.731; MedViTV2-L 0.943 / 0.716. | MedViTV2 Table 5 misquotes 8 TissueMNIST baseline rows (see `decisions.tsv`). | published |
| `zheng2023complexmixer` | MLP-mixer variant on 28 px images. | 0.944 / 0.739 (Table II). | 28 px input. | no |
| `doerrich2026mopet` | Parameter-efficient mixture-of-experts over a frozen DINOv3-B, 3 seeds. | Accuracy 0.7538 ± 0.0013, no per-dataset AUROC (App. Table 0.B2). | Frozen backbone with adapters. | no |
| `woloshuk2021kidney` | Source study of BBBC051: 3D and 2D-projection CNNs on nuclei from 3 donor specimens. | Balanced accuracy 80.26% (3D), 66.55% (2D projection); own 70/15/15 split, 8 classes condensed from 11, context crops. | Not comparable: other split, metric, and inputs. Its cross-specimen test (58.3% balanced accuracy on an unseen specimen) shows the donor shift our cell-level split does not test. | no |

Not comparable or no result: PNW (`deng2026pnw`) reports training-set accuracy only; MambaMIC and GradAttn use their own splits; the shared-backbone paper reports macro-F1 only; UMedPT excludes TissueMNIST; MedMamba, MedMambaLite, HamVision, ZACH-ViT do not use it.

## Not read in full

DFF-Net (doi 10.1049/ipr2.70495, publisher returned 403), SULBA's supplementary tables, and eight paywalled citers of MedViTV2 and MedKAN (MedVK, MedCTM, EfficientMedFormer, TriViT, DyMedNet, and three others listed in the reader report). A result above ours could still hide in one of them.
