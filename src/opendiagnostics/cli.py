"""Command line: train, evaluate, compare, publish, notebook, and list."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from opendiagnostics.config import load_config


def _resolve_plugins(plugins: list[str], config_dir: Path) -> list[str]:
    # Absolute paths let the config saved in a run directory still find file plugins.
    return [str((config_dir / p).resolve()) if p.endswith(".py") else p for p in plugins]


def _train(args: argparse.Namespace) -> int:
    from opendiagnostics.engine import train

    config = load_config(args.config, args.overrides)
    config_dir = args.config.resolve().parent
    config.plugins = _resolve_plugins(config.plugins, config_dir)
    if args.output_dir:
        config.output_dir = args.output_dir
    print(train(config, config_dir))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from opendiagnostics.engine import evaluate_run

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

    target = publish_run(
        args.run_dir,
        args.project,
        include_predictions=not args.no_predictions,
        allow_dirty=args.allow_dirty,
    )
    print(target)
    return 0


def _notebook(args: argparse.Namespace) -> int:
    from opendiagnostics.notebook import build_notebook

    projects = build_notebook(args.models, args.out)
    runs = sum(len(p.runs) for p in projects)
    print(f"wrote {len(projects)} projects and {runs} runs to {args.out}")
    return 0


def _list(args: argparse.Namespace) -> int:
    from opendiagnostics.registry import DATASETS, MODELS, TASKS, TRANSFORMS, load_all

    load_all()
    for registry in (DATASETS, MODELS, TASKS, TRANSFORMS):
        print(f"{registry.kind}s: {', '.join(registry.names())}")
    return 0


def _parser() -> tuple[argparse.ArgumentParser, dict[str, Callable[[argparse.Namespace], int]]]:
    parser = argparse.ArgumentParser(prog="opendiagnostics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train and evaluate one experiment")
    train.add_argument("config", type=Path, help="experiment YAML file")
    train.add_argument("overrides", nargs="*", help="config overrides such as train.epochs=2")
    train.add_argument("--output-dir", help="where run directories go (overrides output_dir)")

    evaluate = sub.add_parser("evaluate", help="re-evaluate a finished run")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--split", action="append", help="split to evaluate (repeatable)")

    compare = sub.add_parser(
        "compare", help="paired patient-level bootstrap of B minus A on the same test samples"
    )
    compare.add_argument("a", type=Path, help="predictions CSV of the reference model")
    compare.add_argument("b", type=Path, help="predictions CSV of the challenger")
    compare.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    compare.add_argument("--replicates", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--json", action="store_true", help="print the result as JSON")

    publish = sub.add_parser("publish", help="copy a finished run into a model project")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument("project", type=Path, help="model project, e.g. models/pneumoniamnist-cnn")
    publish.add_argument(
        "--no-predictions",
        action="store_true",
        help="leave out per-sample predictions (when the dataset license forbids sharing them)",
    )
    publish.add_argument("--allow-dirty", action="store_true", help="publish a dirty-tree run")

    notebook = sub.add_parser("notebook", help="generate the lab notebook pages for the docs")
    notebook.add_argument("--models", type=Path, default=Path("models"))
    notebook.add_argument("--out", type=Path, default=Path("docs/notebook"))

    sub.add_parser("list", help="show registered datasets, models, tasks, and transforms")
    commands = {
        "train": _train,
        "evaluate": _evaluate,
        "compare": _compare,
        "publish": _publish,
        "notebook": _notebook,
        "list": _list,
    }
    return parser, commands


def main(argv: list[str] | None = None) -> int:
    parser, commands = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
