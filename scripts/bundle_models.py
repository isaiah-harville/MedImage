"""Assemble the inference service's model folder from published runs.

    uv run python scripts/bundle_models.py OUT [--models models]

For each project under ``models/`` whose latest published run has a checkpoint URL, that run
is bundled into ``OUT/<project>/``: its ``config.yaml``, ``metrics.json``, ``artifacts.yaml``,
license-cleared ``examples/``, and a ``RUN_ID`` file. Checkpoints are not bundled; the service
downloads them with ``opendiagnostics-fetch-weights``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

CHECKPOINT = "checkpoint.ckpt"
NOTHING_PUBLISHED = 3


def _has_checkpoint_url(run: Path) -> bool:
    files = (yaml.safe_load((run / "artifacts.yaml").read_text()) or {}).get("files", [])
    return any(f["file"] == CHECKPOINT and f.get("url") for f in files)


def bundle(project: Path, out: Path) -> Path | None:
    published = sorted(p for p in (project / "runs").glob("*") if p.is_dir())
    if not published:
        return None
    run = published[-1]
    if not _has_checkpoint_url(run):
        print(f"skipped {project.name}: {run.name} has no public checkpoint", file=sys.stderr)
        return None
    target = out / project.name
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for file in ("config.yaml", "metrics.json", "artifacts.yaml"):
        shutil.copyfile(run / file, target / file)
    if (run / "examples").is_dir():
        shutil.copytree(run / "examples", target / "examples")
    (target / "RUN_ID").write_text(run.name + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    bundled = []
    for project in sorted(
        p for p in args.models.iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        target = bundle(project, args.out)
        if target is not None:
            bundled.append(target.name)
            print(f"bundled {target.name}")
    for stale in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name not in bundled):
        shutil.rmtree(stale)
        print(f"removed {stale.name}, which has no published run")
    if not bundled:
        print(f"no published runs with public checkpoints under {args.models}", file=sys.stderr)
        return NOTHING_PUBLISHED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
