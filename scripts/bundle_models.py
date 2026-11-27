"""Assemble the inference service's weights folder from published runs.

    uv run python scripts/bundle_models.py OUT [--runs runs] [--models models]

For each project under ``models/`` with a published run, the latest run is bundled into
``OUT/<project>/``: its ``config.yaml``, ``metrics.json``, ``artifacts.yaml``, license-cleared
``examples/``, a ``RUN_ID`` file, and ``checkpoint.ckpt``. The checkpoint comes from ``--runs``
(the local training output) when it is there, else from its public URL in ``artifacts.yaml``.
Its SHA-256 must match ``artifacts.yaml``. A checkpoint already in ``OUT`` with the right
SHA-256 is kept, so a cached ``OUT`` downloads only what changed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

CHECKPOINT = "checkpoint.ckpt"
NOTHING_PUBLISHED = 3
# Cloudflare, in front of the weights host, refuses Python's default User-Agent.
USER_AGENT = "opendiagnostics-bundle"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(run: Path) -> dict:
    files = (yaml.safe_load((run / "artifacts.yaml").read_text()) or {}).get("files", [])
    matches = [f for f in files if f["file"] == CHECKPOINT]
    if not matches:
        raise ValueError(f"{run} lists no {CHECKPOINT} in artifacts.yaml")
    return matches[0]


def bundle(project: Path, out: Path, runs: Path) -> Path | None:
    published = sorted(p for p in (project / "runs").glob("*") if p.is_dir())
    if not published:
        return None
    run = published[-1]
    artifact = _artifact(run)
    name = yaml.safe_load((run / "config.yaml").read_text())["name"]
    local = runs / name / run.name / CHECKPOINT
    target = out / project.name
    checkpoint = target / CHECKPOINT
    if not (checkpoint.is_file() and _sha256(checkpoint) == artifact["sha256"]):
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        if local.is_file():
            shutil.copyfile(local, checkpoint)
        elif artifact.get("url"):
            with (
                urllib.request.urlopen(
                    urllib.request.Request(artifact["url"], headers={"User-Agent": USER_AGENT}),
                    timeout=60,
                ) as response,
                checkpoint.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle, length=1 << 20)
        else:
            shutil.rmtree(target)
            raise FileNotFoundError(
                f"{project.name}: no checkpoint at {local} and no URL in {run / 'artifacts.yaml'}"
            )
        if _sha256(checkpoint) != artifact["sha256"]:
            shutil.rmtree(target)
            raise ValueError(f"{project.name}: checkpoint SHA-256 does not match artifacts.yaml")
    shutil.rmtree(target / "examples", ignore_errors=True)
    for file in ("config.yaml", "metrics.json", "artifacts.yaml"):
        shutil.copyfile(run / file, target / file)
    if (run / "examples").is_dir():
        shutil.copytree(run / "examples", target / "examples")
    (target / "RUN_ID").write_text(run.name + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    bundled = []
    for project in sorted(
        p for p in args.models.iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        target = bundle(project, args.out, args.runs)
        if target is not None:
            bundled.append(target.name)
            print(f"bundled {target.name}")
    if not bundled:
        print(f"no published runs under {args.models}", file=sys.stderr)
        return NOTHING_PUBLISHED
    for stale in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name not in bundled):
        shutil.rmtree(stale)
        print(f"removed {stale.name}, which has no published run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
