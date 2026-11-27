# Related work

The closest work, from the ledger in `sources.tsv`. "Ours" is the ConvNeXt-S run: test AUROC 0.9865 [0.9809, 0.9909], accuracy 0.9177, on the official 224-pixel DermaMNIST split.

## Closest set

| Key | What it does | How this result differs | Compared in our experiments? |
|---|---|---|---|
| `wu2025rethinking` | Fine-tunes 12 pretrained encoders end to end and as linear probes on MedMNIST at 224, 3 runs each. On DermaMNIST, DINOv2 ViT-B/14 end to end reaches AUC 98.92 ± 0.11 and ACC 92.02 ± 0.32; CLIP, ViT-B/16, OpenAI, and DINO ViT-B/16 all reach 98.0 to 98.3 (Table 1). It attributes MedMNIST+'s lower end-to-end numbers to MedMNIST+'s training pipeline (Sec. 3.2.4). | Our AUROC is lower than its best and its best lies inside our interval. Its finding that the MedMNIST+ pipeline understates end-to-end fine-tuning is the explanation our draft proposed as a hypothesis, already shown with the same architectures. Its encoder learning rate was picked per model from three values, and the paper does not say whether validation or test data chose it. | No. It is missing from `baselines.yaml` and from the old paper. |
| `sanap2026beyond` | Trains four ImageNet backbones on native 224 DermaMNIST with no augmentation, one seed, and ensembles them. ConvNeXt-Base AUC 0.9820, ViT-Base 0.9821, soft voting 0.9877, ACC 90.92; stacking ACC 91.67 (Table 6). | Single ConvNeXt-Base AUC equals ours within our interval. The soft-voting ensemble is above our point estimate. Whether its AUC is macro or weighted is ambiguous. | No. Missing from `baselines.yaml`. |
| `doerrich2025rethinking` | MedMNIST+: ten networks, three training schemes, four resolutions, one shared setup with no augmentation and early stopping on validation loss. Best DermaMNIST 224 AUC 96.83 (DINO ViT-B/16 linear probe, Table S4). | Our interval lies above all its DermaMNIST values. That comparison changes five or more factors at once. | Yes, three rows in `baselines.yaml`. |
| `abhishek2025investigating` | Audits DermaMNIST: the MedMNIST split ignored lesion IDs, 1,006 of 7,470 lesions span partitions, 886 images overlap train and test. Releases DermaMNIST-C (lesion-disjoint) and DermaMNIST-E, and the index-to-image mapping. ResNet-18/50 on DermaMNIST-C 224: AUC 0.953/0.948. | We used its mapping to score our run on test images whose lesion is not in training (`analysis/lesion_overlap.py`): AUROC 0.9795 [0.9674, 0.9883] on 1,312 images against 0.9912 on the 693 that share a lesion with training. Our training set was the original one, so this is not a DermaMNIST-C result. | Not a baseline. It is the reason the official-split comparison is weaker than it looks. |
| `yang2023medmnist` | MedMNIST v2: the DermaMNIST release, split 7:1:2, ResNet and AutoML baselines on 28-pixel images (224 rows are upscaled 28). Best AUC 0.920. | Different input; our row is like-for-like only with native-224 work. | Yes, in `baselines.yaml`. |
| `manzari2023medvit`, `manzari2025medical` | MedViT and MedViTV2 on MedMNIST v2 settings with resized images. Best DermaMNIST AUC 0.937 and 0.950. | Lower; not native 224. No seeds or intervals. | Yes, in `baselines.yaml`. |
| `zhang2026weighted` | ResNet-18 with weighted sampling and attention on native 224 DermaMNIST, 3 seeds. Best AUC 0.978 ± 0.001 (focal loss), vanilla 0.973. | Below our interval's lower bound by 0.003 at best. | No. |
| `baek2026detection` | Seed-controlled audit of augmentation gains on MedMNIST+ at 128 pixels. On DermaMNIST, standard augmentation adds about 0.017 to 0.022 AUC for ResNets from scratch; an exploratory pretrained ViT-S/16 reaches 0.981. Seed SD of ResNet-18 AUC 0.0037. | Supports augmentation as one cause of the gap to MedMNIST+, at 128 pixels and from archive tables, not a paper. Gives a seed-noise estimate for DermaMNIST. | No. |

## Other sources read

`kadric2025improving` (DermaMNIST-C 224, AUC 0.9768, frozen backbones), `disalvo2024medmnistc` (corruption benchmark), `doerrich2026mopet` (LoRA experts on DINOv3, ACC 88.96), `huang2026medmnist` (ReScience replication of MedMNIST v2 ResNets on DermaMNIST), `shehu2026litemedvit`, `gavril2026shared`, `amangeldi2025cnn`, `medeiros2026believe` (TTA at 28 pixels), `deng2026benchmarking` (100% is training accuracy, not test), `elkholy2026comparative` (abstract only). None reports a DermaMNIST 224 result at our level on the official test split.

## Position

A prior paper already reports this result. `wu2025rethinking` fine-tunes ViTs end to end on the same 224-pixel images and split and reports a mean of three runs, 0.9892, inside our interval, plus four more models between 0.980 and 0.983. It also already argues that the MedMNIST+ pipeline, not the architectures, explains MedMNIST+'s lower end-to-end numbers. The old draft's statement that our interval "lies above every published DermaMNIST value we could source" is false.

What no source in the ledger reports: per-image predictions or bootstrap intervals on DermaMNIST, and any measurement of how much lesion overlap inflates a model's official-split score.
