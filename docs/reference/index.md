# Develop

Everything here is for training, evaluating, and publishing models with the pipeline.

- [Using the pipeline](../guide.md) takes one experiment from a config to a published result.
- [Model weights](../weights.md) covers downloading, checking, loading, and publishing
  checkpoints.
- [Design](../design.md) explains how the pipeline works and why: data contracts, evaluation
  rules, and where artifacts live.

## Module reference

The module pages are generated from the docstrings in `src/opendiagnostics`. Training runs
through Lightning's `LightningCLI`; start with the [command line](cli.md), then the
[data modules](data.md) and [Lightning modules](modules.md).

- [Command line](cli.md)
- [Data modules and splits](data.md)
- [Lightning modules](modules.md)
- [Evaluation, examples, and provenance](evaluation.md)
- [Metrics](metrics.md)
- [Publishing runs](publish.md)
- [Lab notebook generator](notebook.md)
- [Inference service](serve.md)
