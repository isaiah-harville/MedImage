# OpenDiagnostics

**Open research models for medical image diagnosis. Not a medical device.**

OpenDiagnostics is an open-source monorepo of PyTorch models for medical image
classification and detection. The goal is models that beat published results, with the
statistics to back the claim.

**[Try the demo](https://diagnostics.harville.dev).** Pick a scan type, run a published
model on a sample image or your own, and see its prediction with a map of the pixels that
drove it. Uploads are never stored.

- [Lab notebook](notebook/index.md): every model project, its leaderboard against published
  baselines, and every run with its metrics, 95% patient-level intervals, and weights.
- [Datasets](datasets.md): the public datasets we survey, their licenses and access terms,
  and the bar each model has to beat.
- [Code reference](reference/index.md): how to use the pipeline, the design, the model
  weights, and every module.
