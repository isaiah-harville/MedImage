"""Copy a finished run into a model project so the lab notebook can show it.

A published run is ``models/<project>/runs/<run-id>/``. It holds the small files from the run
directory, which are committed to git and served by the docs site, and ``artifacts.yaml``,
which lists the large files (the checkpoint) by size and SHA-256. The checkpoint itself is
committed through Git LFS when the dataset's license allows sharing trained weights, and so
are the example figures (``examples/``); otherwise neither is copied.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

SMALL_FILES = ("config.yaml", "run.json", "metrics.json", "metrics.csv", "splits.json")
LARGE_FILES = ("checkpoint.ckpt",)
MAX_SMALL_FILE_BYTES = 20 * 1024 * 1024
EXAMPLES = "examples"
EXAMPLES_MANIFEST = "examples.json"
# Licenses that allow redistributing a dataset's images and models trained on it, so example
# figures and checkpoints may be committed. Anything else, including credentialed data and
# unknown licenses, publishes neither.
OPEN_LICENSES = frozenset({"CC0 1.0", "CC BY 3.0", "CC BY 4.0", "CC BY-NC 4.0", "generated"})


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


def _license_allows_sharing(run_dir: Path) -> bool:
    record = json.loads((run_dir / "run.json").read_text())
    return record.get("dataset", {}).get("license") in OPEN_LICENSES


def publish_run(
    run_dir: Path,
    project_dir: Path,
    include_predictions: bool = True,
    allow_dirty: bool = False,
    include_checkpoint: bool = True,
) -> Path:
    """Copy a finished run into ``project_dir/runs/<run-id>``.

    The small files are committed as they are. The checkpoint is copied too, for Git LFS
    (see ``.gitattributes``), when the dataset's license allows sharing trained weights and
    ``include_checkpoint`` is true; ``artifacts.yaml`` records its size and SHA-256 either
    way. Rerunning on the same run gives the same result and keeps any URLs already recorded.
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

    examples = run_dir / EXAMPLES
    shutil.rmtree(target / EXAMPLES, ignore_errors=True)
    if include_predictions and examples.is_dir() and _license_allows_sharing(run_dir):
        shutil.copytree(examples, target / EXAMPLES)

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
        committed = target / name
        if include_checkpoint and _license_allows_sharing(run_dir):
            if not committed.exists() or sha256(committed) != digest:
                shutil.copyfile(source, committed)
        else:
            committed.unlink(missing_ok=True)
    (target / "artifacts.yaml").write_text(
        yaml.safe_dump(Artifacts(files=artifacts).model_dump(), sort_keys=False)
    )
    return target
