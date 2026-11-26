"""Command line: ``medimage train``, ``medimage evaluate``, ``medimage list``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from medimage.config import load_config


def _resolve_plugins(plugins: list[str], config_dir: Path) -> list[str]:
    # Make file plugins absolute so the saved config still finds them from the run directory.
    return [str((config_dir / p).resolve()) if p.endswith(".py") else p for p in plugins]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medimage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train and evaluate one experiment")
    train.add_argument("config", type=Path, help="experiment YAML file")
    train.add_argument("overrides", nargs="*", help="config overrides such as train.epochs=2")
    train.add_argument("--output-dir", help="where run directories go (overrides output_dir)")

    evaluate = sub.add_parser("evaluate", help="re-evaluate a finished run")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--split", action="append", help="split to evaluate (repeatable)")

    sub.add_parser("list", help="show registered datasets, models, tasks, and transforms")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )

    if args.command == "train":
        from medimage.engine import train as run_train

        config = load_config(args.config, args.overrides)
        config_dir = args.config.resolve().parent
        config.plugins = _resolve_plugins(config.plugins, config_dir)
        if args.output_dir:
            config.output_dir = args.output_dir
        run_dir = run_train(config, config_dir)
        print(run_dir)
        return 0

    if args.command == "evaluate":
        from medimage.engine import evaluate_run

        results = evaluate_run(args.run_dir, args.split)
        print(json.dumps(results, indent=2, default=str))
        return 0

    from medimage.registry import DATASETS, MODELS, TASKS, TRANSFORMS, load_all

    load_all()
    for registry in (DATASETS, MODELS, TASKS, TRANSFORMS):
        print(f"{registry.kind}s: {', '.join(registry.names())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
