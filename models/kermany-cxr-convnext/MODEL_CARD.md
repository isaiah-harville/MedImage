# Model card: kermany-cxr-convnext

## Summary
ConvNeXt-S, pre-trained on ImageNet-22k, fine-tuned at 448 px to tell pneumonia from a normal
pediatric chest X-ray. No run is published yet; the headline metric will be AUROC on the
official 624-image test folder, with a patient-level 95% bootstrap interval.

## Intended use
Research only. Not a medical device. It must not be used to diagnose pneumonia in any child.

## Data
- Kermany D, Zhang K, Goldbaum M, "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
  Images for Classification", Mendeley Data v3, doi:10.17632/rscbjbr9sj.3. Cite Kermany et
  al., Cell 172(5):1122-1131.e9, 2018.
- License: CC BY 4.0. The Mendeley page also says the data is for research use only, which is
  not a CC BY term.
- `scripts/datasets/kermany_cxr.py` downloads `ZhangLabData.zip` (SHA-256 checked), unpacks
  `CellData/chest_xray`, and writes `manifest.csv`. The patient ID is the number in each
  filename, `<CLASS>-<patient>-<n>.jpeg`; bacterial and viral pneumonia are both labelled
  PNEUMONIA.
- Split: the official test folder (624 images: 234 normal, 390 pneumonia) is the test split;
  no patient in it is in the training folder. Validation is fold 0 of 10 patient-grouped,
  label-stratified folds of the training folder (5,232 images).

## Training
Config: `config.yaml`. AdamW, cosine schedule over 30 epochs, bf16 mixed precision, EMA
weights, batches of 16 with four accumulated per optimizer step. Augmentation is a random
resized crop and small affine jitter, with no flips. The decision threshold is fitted on
validation (Youden's J).

## Results
No published run yet.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- Pediatric images only, so expect shift on adults and on other sites.
- The test set is 62.5% pneumonia (390 of 624), so PPV and NPV at our threshold will not carry over to another population.
- No published baseline is listed yet; see `baselines.yaml`.

## Reproduce
```bash
uv run python scripts/datasets/kermany_cxr.py data/kermany_cxr
uv run opendiagnostics fit -c models/kermany-cxr-convnext/config.yaml
```
