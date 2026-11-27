# Model card: kermany-oct-convnext

## Summary
ConvNeXt-S, pre-trained on ImageNet-22k, fine-tuned at 448 px to classify a retinal OCT
B-scan as CNV, DME, drusen or normal. No run is published yet; the headline metric will be
accuracy on the official 1,000-image test folder, with a patient-level 95% bootstrap interval.

## Intended use
Research only. Not a medical device. It must not be used to diagnose or refer patients.

## Data
- Kermany D, Zhang K, Goldbaum M, "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
  Images for Classification", Mendeley Data v3, doi:10.17632/rscbjbr9sj.3. Cite Kermany et
  al., Cell 172(5):1122-1131.e9, 2018.
- License: CC BY 4.0. The Mendeley page also says the data is for research use only, which is
  not a CC BY term.
- `scripts/datasets/kermany_oct.py` downloads `ZhangLabData.zip` (SHA-256 checked), unpacks
  `CellData/OCT`, and writes `manifest.csv`. The patient ID is the number in each filename,
  `<CLASS>-<patient>-<n>.jpeg`.
- Split: the official test folder (1,000 images, 250 per class) is the test split; no
  patient in it is in the training folder. Validation is fold 0 of 10 patient-grouped,
  label-stratified folds of the training folder (108,309 images).
- Labels come from tiered graders of increasing expertise.

## Training
Config: `config.yaml`. AdamW, cosine schedule over 10 epochs, bf16 mixed precision, EMA
weights, label smoothing 0.1, batches of 16 with four accumulated per optimizer step.
Augmentation is a random resized crop and horizontal and vertical flips; evaluation averages
each scan with its horizontal flip. The checkpoint with the best validation accuracy is kept.

## Results
No published run yet.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- One device family (Heidelberg Spectralis SD-OCT), so expect shift on other scanners.
- The balanced test set does not reflect clinical prevalence, and one image is 0.1 accuracy
  points, so published results near 99.8% differ by a few images.
- 787 patient IDs appear under more than one class, so one patient's images can carry
  different labels.

## Reproduce
```bash
uv run python scripts/datasets/kermany_oct.py data/kermany_oct
uv run opendiagnostics fit -c models/kermany-oct-convnext/config.yaml
```
