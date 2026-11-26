"""Copy a finished run into a model project so the lab notebook can show it.

A published run is ``models/<project>/runs/<run-id>/``. It holds the small files from the run
directory, which are committed to git and served by the docs site, and ``artifacts.yaml``,
which lists the large files (the checkpoint) by size, SHA-256, and download URL. Large
files are uploaded to external storage separately; see docs/design.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

SMALL_FILES = ("config.yaml", "run.json", "metrics.json", "history.jsonl", "splits.json")
LARGE_FILES = ("checkpoint.pt",)
MAX_SMALL_FILE_BYTES = 20 * 1024 * 1024


class PublishError(RuntimeError):
    """Raised when a run cannot be published as asked."""


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    bytes: int
    sha256: str
    url: str | None = None


class Artifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[Artifact]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts(path: Path) -> Artifacts:
    if not path.exists():
        return Artifacts(files=[])
    return Artifacts.model_validate(yaml.safe_load(path.read_text()) or {"files": []})


def publish_run(
    run_dir: Path,
    project_dir: Path,
    include_predictions: bool = True,
    allow_dirty: bool = False,
) -> Path:
    """Copy ``run_dir``'s small files into ``project_dir/runs/<run-id>`` and list its large files.

    Rerunning on the same run gives the same result and keeps any URLs already recorded.
    """
    run_dir, project_dir = Path(run_dir), Path(project_dir)
    if not (run_dir / "metrics.json").exists():
        raise PublishError(f"{run_dir} has no metrics.json; publish a run that finished")
    if not (project_dir / "config.yaml").exists():
        raise PublishError(f"{project_dir} is not a model project (no config.yaml)")
    git = json.loads((run_dir / "run.json").read_text()).get("git", {})
    if git.get("dirty") and not allow_dirty:
        raise PublishError(
            f"{run_dir.name} ran on a dirty working tree, so its commit does not reproduce it. "
            "Rerun on a clean commit, or pass --allow-dirty to publish it anyway."
        )

    target = project_dir / "runs" / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    names = list(SMALL_FILES)
    if include_predictions:
        names += sorted(p.name for p in run_dir.glob("predictions_*.csv"))
    for name in names:
        source = run_dir / name
        if not source.exists():
            raise PublishError(f"{source} is missing")
        if source.stat().st_size > MAX_SMALL_FILE_BYTES:
            raise PublishError(f"{source} is over 20 MB; host it as a large artifact instead")
        shutil.copyfile(source, target / name)

    known = {a.file: a for a in load_artifacts(target / "artifacts.yaml").files}
    artifacts = []
    for name in LARGE_FILES:
        source = run_dir / name
        if not source.exists():
            continue
        digest = sha256(source)
        previous = known.get(name)
        url = previous.url if previous is not None and previous.sha256 == digest else None
        artifacts.append(Artifact(file=name, bytes=source.stat().st_size, sha256=digest, url=url))
    (target / "artifacts.yaml").write_text(
        yaml.safe_dump(Artifacts(files=artifacts).model_dump(), sort_keys=False)
    )
    return target
