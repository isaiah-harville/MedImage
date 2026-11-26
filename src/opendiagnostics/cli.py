"""Command line.

``opendiagnostics fit -c CONFIG [--trainer.max_epochs=2 ...]`` trains with Lightning's
``LightningCLI``, then evaluates the selected checkpoint with bootstrap intervals. The other
commands work on finished runs: ``evaluate``, ``compare``, ``publish``, and ``notebook``.

A run directory holds everything needed to audit or repeat an experiment::

    runs/<name>/<timestamp>-<commit>/
        config.yaml          the fully resolved config
        run.json             seed, git commit and dirty flag, package versions, hardware
        splits.json          sample IDs in each split, with patient and label counts
        metrics.csv          one row per epoch from Lightning's CSVLogger
        checkpoint.ckpt      weights of the epoch with the best validation monitor
        metrics.json         final metrics with bootstrap confidence intervals
        predictions_<split>.csv
        train.log
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import CSVLogger

from opendiagnostics import provenance
from opendiagnostics.data import MedicalDataModule, summarize
from opendiagnostics.evaluation import EvalSettings, evaluate, write_json
from opendiagnostics.modules import DiagnosticModule

log = logging.getLogger("opendiagnostics")

# run.json and splits.json are written before the first metric, so these always fire.
warnings.filterwarnings("ignore", message=r".*directory .* exists and is not empty")


def _version(output_dir: str, name: str, commit: str | None) -> str:
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{(commit or 'nogit')[:8]}"
    version, suffix = stamp, 1
    while (Path(output_dir) / name / version).exists():
        version, suffix = f"{stamp}-{suffix}", suffix + 1
    return version


class OpenDiagnosticsCLI(LightningCLI):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("seed_everything_default", 0)
        kwargs.setdefault("save_config_kwargs", {"config_filename": "config.yaml"})
        defaults = {"deterministic": True, "num_sanity_val_steps": 0}
        kwargs["trainer_defaults"] = defaults | kwargs.get("trainer_defaults", {})
        super().__init__(
            DiagnosticModule,
            MedicalDataModule,
            *args,
            subclass_mode_model=True,
            subclass_mode_data=True,
            **kwargs,
        )

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        return {"fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"}}

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--name", type=str, required=True, help="run name; runs go to output_dir/name"
        )
        parser.add_argument("--output_dir", type=str, default="runs")
        parser.add_argument("--notes", type=str, default="")
        parser.add_class_arguments(EvalSettings, "eval")
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                "checkpoint.monitor": "val/auroc",
                "checkpoint.mode": "max",
                "checkpoint.filename": "checkpoint",
                "checkpoint.save_weights_only": True,
                "checkpoint.auto_insert_metric_name": False,
                "checkpoint.enable_version_counter": False,
            }
        )
        parser.link_arguments("data.info", "model.init_args.info", apply_on="instantiate")

    @property
    def settings(self) -> Any:
        return self.config[self.subcommand] if self.subcommand else self.config

    @property
    def checkpoint(self) -> ModelCheckpoint:
        callback = self.trainer.checkpoint_callback
        assert isinstance(callback, ModelCheckpoint)
        return callback

    def instantiate_trainer(self, **kwargs: Any) -> Any:
        s = self.config_init[self.subcommand] if self.subcommand else self.config_init
        self.git = provenance.git_info(Path.cwd())
        version = _version(s.output_dir, s.name, self.git.get("commit"))
        kwargs["logger"] = CSVLogger(save_dir=s.output_dir, name=s.name, version=version)
        return super().instantiate_trainer(**kwargs)

    def before_fit(self) -> None:
        self.started = provenance.utc_now()
        data = self.datamodule
        data.prepare_data()
        data.setup("fit")
        self.run_dir = Path(self.trainer.log_dir or ".")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint.dirpath = str(self.run_dir)
        self.handler = logging.FileHandler(self.run_dir / "train.log")
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        for name in ("opendiagnostics", "lightning.pytorch"):
            logging.getLogger(name).addHandler(self.handler)
        s = self.settings
        source, info = data.source, data.info
        self.record = {
            "name": s.name,
            "seed": s.seed_everything,
            "deterministic": s.trainer.deterministic,
            "git": self.git,
            "environment": provenance.environment(),
            "device": str(self.trainer.strategy.root_device),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "dataset": {
                "name": type(data).__name__,
                "license": info.license,
                "citation": info.citation,
                "patient_ids_available": info.patient_ids_available,
                "fingerprint": source.fingerprint,
            },
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "started": self.started,
        }
        write_json(self.run_dir / "run.json", self.record)
        write_json(
            self.run_dir / "splits.json",
            {
                "method": data.split,
                "summary": summarize(data.splits, source.patient_ids, source.strata),
                "ids": {k: [source.sample_ids[i] for i in v] for k, v in data.splits.items()},
            },
        )
        log.info("run directory: %s", self.run_dir)

    def after_fit(self) -> None:
        try:
            checkpoint = Path(self.checkpoint.best_model_path)
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"no checkpoint was saved; check that checkpoint.monitor "
                    f"{self.settings.checkpoint.monitor!r} is a logged validation metric"
                )
            s = self.settings
            results = evaluate(
                self.trainer,
                self.model,
                self.datamodule,
                EvalSettings(**vars(s.eval)),
                s.seed_everything,
                checkpoint,
                self.run_dir,
            )
            results["selection"] = {
                "monitor": s.checkpoint.monitor,
                "mode": s.checkpoint.mode,
                "best_epoch": results["checkpoint_epoch"],
            }
            write_json(self.run_dir / "metrics.json", results)
            self.record["finished"] = provenance.utc_now()
            write_json(self.run_dir / "run.json", self.record)
            log.info("finished: %s", self.run_dir / "metrics.json")
            print(self.run_dir)
        finally:
            for name in ("opendiagnostics", "lightning.pytorch"):
                logging.getLogger(name).removeHandler(self.handler)
            self.handler.close()


def fit(args: list[str] | None = None) -> Path:
    """Run ``fit`` with ``args`` (or the command line when ``None``); return the run directory."""
    cli = OpenDiagnosticsCLI(args=None if args is None else ["fit", *args])
    return cli.run_dir


def evaluate_run(run_dir: Path | str, splits: list[str] | None = None) -> dict[str, Any]:
    """Re-evaluate a finished run from its saved config and checkpoint."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    skipped = OpenDiagnosticsCLI.subcommands()["fit"] | {"self"}
    for argument in set(inspect.signature(Trainer.fit).parameters) - skipped:
        config.pop(argument, None)
    if splits:
        config["eval"]["splits"] = splits
    cli = OpenDiagnosticsCLI(args=config, run=False, save_config_callback=None)
    cli.datamodule.prepare_data()
    cli.datamodule.setup("fit")
    seed = cli.config.seed_everything
    return evaluate(
        cli.trainer,
        cli.model,
        cli.datamodule,
        EvalSettings(**vars(cli.config.eval)),
        seed,
        run_dir / "checkpoint.ckpt",
        run_dir,
    )


def _evaluate(args: argparse.Namespace) -> int:
    print(json.dumps(evaluate_run(args.run_dir, args.split), indent=2, default=str))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from opendiagnostics.metrics.compare import compare, read_predictions

    result = compare(
        read_predictions(args.a),
        read_predictions(args.b),
        metric=args.metric,
        replicates=args.replicates,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    floor = 1 / result.valid_replicates if result.valid_replicates else float("nan")
    p = f"< {floor:.4f}" if result.p_value == 0 else f"{result.p_value:.4f}"
    print(
        f"{result.metric}: A {result.a:.4f}, B {result.b:.4f}\n"
        f"B - A = {result.difference:+.4f} "
        f"[{round(result.confidence * 100)}% CI {result.lower:+.4f}, {result.upper:+.4f}], p {p}\n"
        f"{result.samples} samples from {result.patients} patients, "
        f"{result.valid_replicates} of {result.replicates} replicates valid"
    )
    return 0


def _publish(args: argparse.Namespace) -> int:
    from opendiagnostics.publish import publish_run

    print(
        publish_run(
            args.run_dir,
            args.project,
            include_predictions=not args.no_predictions,
            allow_dirty=args.allow_dirty,
        )
    )
    return 0


def _notebook(args: argparse.Namespace) -> int:
    from opendiagnostics.notebook import build_notebook

    projects = build_notebook(args.models, args.out)
    print(
        f"wrote {len(projects)} projects and {sum(len(p.runs) for p in projects)} runs to {args.out}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opendiagnostics", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "fit", help="train and evaluate one experiment (see `opendiagnostics fit --help`)"
    )

    evaluate = sub.add_parser("evaluate", help="re-evaluate a finished run")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--split", action="append", help="split to evaluate (repeatable)")
    evaluate.set_defaults(handler=_evaluate)

    compare = sub.add_parser(
        "compare", help="paired patient-level bootstrap of B minus A on the same test samples"
    )
    compare.add_argument("a", type=Path, help="predictions CSV of the reference model")
    compare.add_argument("b", type=Path, help="predictions CSV of the challenger")
    compare.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    compare.add_argument("--replicates", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--json", action="store_true", help="print the result as JSON")
    compare.set_defaults(handler=_compare)

    publish = sub.add_parser("publish", help="copy a finished run into a model project")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument(
        "project", type=Path, help="model project, e.g. models/pneumoniamnist-convnext"
    )
    publish.add_argument(
        "--no-predictions",
        action="store_true",
        help="leave out per-sample predictions (when the dataset license forbids sharing them)",
    )
    publish.add_argument("--allow-dirty", action="store_true", help="publish a dirty-tree run")
    publish.set_defaults(handler=_publish)

    notebook = sub.add_parser("notebook", help="generate the lab notebook pages for the docs")
    notebook.add_argument("--models", type=Path, default=Path("models"))
    notebook.add_argument("--out", type=Path, default=Path("docs/notebook"))
    notebook.set_defaults(handler=_notebook)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    command = sys.argv[1:] if argv is None else argv
    if command and command[0] == "fit":
        fit(None if argv is None else argv[1:])
        return 0
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
