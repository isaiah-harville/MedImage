# Model card: cbis-ddsm-fasterrcnn

## Summary
A torchvision Faster R-CNN (ResNet-50 FPN v2, ImageNet backbone) that draws boxes around
masses and calcifications on full screening mammograms. No run is published yet.

## Intended use
Research only. Not a medical device. It must not be used to screen or diagnose patients.

## Data
- CBIS-DDSM on TCIA (doi:10.7937/K9/TCIA.2016.7O02S9CY). Lee et al., Sci Data 4:170177,
  2017. `scripts/datasets/cbis_ddsm.py` downloads the four case-description CSVs and, through
  TCIA's public NBIA API, only the full-mammogram and ROI-mask series they name.
- License: CC BY 3.0, plus the TCIA Data Usage Policy.
- Split: the official train and test cases. Mass and calcification cases were split
  separately, so the script drops training mammograms of participants who also appear in
  test. Validation is fold 0 of 5 patient-grouped folds of the training split.
- Patients: the `P_xxxxx` participant, since one participant has several subject IDs.
- Labels: one box per ROI mask, the mask chosen as the binary ROI image with the
  mammogram's size. A film listed under both mass and calcification cases is kept once, with
  both kinds of box, when the two files hold the same pixels. Pathology is kept in the
  manifest but not used. Mass masks were refined automatically; calcification ROIs are coarse.

## Training
Config: `config.yaml`. Images resized so the short side is 1,333 px (long side at most
2,048), batch 2, 30 epochs of SGD with cosine decay, horizontal and vertical flips. Sized for
a 12 GB GPU; peak memory has not been measured.

## Results
Headline metric: test mAP@0.5 over mass and calcification with a 95% patient-level bootstrap
interval. No published number shares this protocol; see `baselines.yaml`.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- The images are digitised film from the 1990s, not digital mammography; results do not
  transfer to modern screening.
- Calcification boxes cover coarse ROIs, so their mAP understates or misstates localisation.
- Every training image has a lesion; the model has not seen normal mammograms.

## Reproduce
```bash
uv run python scripts/datasets/cbis_ddsm.py --delete-archives
uv run opendiagnostics fit -c models/cbis-ddsm-fasterrcnn/config.yaml
```
