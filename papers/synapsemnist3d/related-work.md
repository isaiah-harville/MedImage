# Related work

Stage 2 stopped here. A prior paper already reports this result (see the first row), so the paper was not outlined or drafted.

## Prior report of the same result

`nuwagira2026topocam` (TopoCAM, ML4H 2025, PMLR 297) reports a 3D ResNet-18 (R3D-18, Kinetics-400 pretrained, fine-tuned) at 64³ on the official SynapseMNIST3D split with AUC 98.5 and accuracy 92.9 (Table 2, Table 9). Our run: AUROC 0.9845 [0.9702, 0.9951], accuracy at 0.5 of 0.9631 [0.9403, 0.9830]. The AUROC is the same to three decimals. The architecture family and resolution are the same. The differences are that our network is trained from scratch, and that we report an interval and release predictions. TopoCAM gives no seeds, intervals, or predictions, and its Table 9 repeats some rows across datasets (the R(2+1)D-18 Adrenal3D row equals its Synapse3D row), so its tables need care.

## Closest set

| Key | What it does | How our run differs | Compared in our experiments? |
|---|---|---|---|
| nuwagira2026topocam | Fine-tunes Kinetics-pretrained R3D-18, MC3-18, R(2+1)D-18 at 64³; TopoCAM adds Grad-CAM ROI Betti curves and an MLP. Synapse3D AUC: R3D-18 98.5, TopoCAM 98.4, R(2+1)D-18 97.5, MC3-18 96.8, M3T 95.8. From-scratch EfficientNet3D, 3D-CCT, 3D-ViT: 49.9 to 58.8. | Same network family and resolution, trained from scratch. Same AUROC. | No. Published value only; no predictions for a paired test. |
| yu2026planecycle | Lifts DINOv3 2D ViTs to 3D; full fine-tuning. Synapse AUC 97.3 (3D ViT-L/16) and 97.1 (PlaneCycle), mean of 5 runs, official split. Resolution not stated in the paper (64³ per code defaults). | 3D CNN from scratch, 33 M parameters, no pretraining. Our 0.9845 is above, but 0.973 lies inside our interval. | No. |
| liu2026manifold | MTDL, Hodge-decomposition channels into a small network, 64³, official split, 3 seeds. Synapse AUC 0.951. | Raw voxels into a standard 3D CNN. Our lower bound 0.9702 is above 0.951. | No. |
| wachira2026combinatorial | CNMTDL, combinatorial-complex extension of MTDL, 64³, 5 seeds. Synapse AUC 0.932. | As above. | No. |
| park2026axial | Frozen MedDINOv3 slice features with cross-plane attention, 64³ resized to 224 in-plane. Synapse AUC 88.0. Compares against 28³ baselines. | End-to-end 3D training. | Listed in baselines.yaml. |
| zhemchuzhnikov2024ilponet | Rotation-invariant ILPO operators, 28³, official split, single run. Synapse AUC 0.940. | Higher resolution, standard convolutions. | No. |
| yang2023medmnist | The benchmark. ResNet-18/50 with 2.5D, 3D, ACS convolutions at 28³, mean of at least 3 trials. Best Synapse AUC 0.851. | 64³ inputs, augmentation, AdamW, cosine, EMA, flip TTA. | Listed in baselines.yaml. |
| zhu2024bsda | BSDA on a 64³ 3D ResNet-18, 3 seeds. No SynapseMNIST3D result. | Not applicable to this subset. | No. |

## Sources not fully read

- `kus2026nas` (PBC-NAS, Springer): abstract only; it covers SynapseMNIST3D but the per-dataset numbers were behind a challenge page.
- `akbacak2024efficient` and `rajoria2024lightweight`: abstract only; both claim to beat the 28³ baselines on SynapseMNIST3D without visible numbers.
- `abe2026sulba`: per-dataset SynapseMNIST3D AUROC is in supplementary tables S11 to S16, not read.
- SKTP-HOPE (TechRxiv 2026): behind a Cloudflare challenge; no row.
