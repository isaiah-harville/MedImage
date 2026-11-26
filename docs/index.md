# MedImage

MedImage is an open-source monorepo of PyTorch models for medical image classification and
detection. The goal is models that beat published results, shown with the statistics to
back the claim.

This site is the project's lab notebook. Every published run has a page with its config,
metrics with 95% patient-level bootstrap intervals, training curves, provenance down to the
git commit, and its files to download.

- [Using the pipeline](guide.md): from a config to a published, comparable result.
- [Lab notebook](notebook/index.md): every model project, its leaderboard against published
  baselines, and every run.
- [Datasets](datasets.md): the public datasets we survey, their licenses and access terms,
  and the shortlist we train on first.
- [Design](design.md): how the pipeline works and why, including where artifacts are hosted.
- [Code reference](reference/index.md): every module, generated from the docstrings.

These models are research artifacts and are not medical devices.
