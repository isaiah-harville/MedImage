"""Copy a finished run into a model project so the lab notebook can show it.

A published run is ``models/<project>/runs/<run-id>/``. It holds the small files from the run
directory, which are committed to git and served by the docs site, and ``artifacts.yaml``,
which lists the large files (the checkpoint) by size, SHA-256, and download URL. When the
dataset's license allows sharing trained weights, the checkpoint is uploaded to the public
weights bucket (see ``docs/weights.md``) and the example figures (``examples/``) are copied;
otherwise neither leaves the machine.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

import boto3
import yaml
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict

SMALL_FILES = ("config.yaml", "run.json", "metrics.json", "metrics.csv", "splits.json")
LARGE_FILES = ("checkpoint.ckpt",)
MAX_SMALL_FILE_BYTES = 20 * 1024 * 1024
EXAMPLES = "examples"
EXAMPLES_MANIFEST = "examples.json"
WEIGHTS_BUCKET = "opendiagnostics-weights"
WEIGHTS_URL = f"https://diagnostics.harville.dev/{WEIGHTS_BUCKET}"
# Cloudflare, in front of the weights host, refuses Python's default User-Agent.
USER_AGENT = "opendiagnostics"
# Licenses that allow redistributing a dataset's images and models trained on it, so example
# figures and checkpoints may be published. Anything else, including credentialed data and
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


Upload = Callable[[Path, str], None]


def upload_weights(path: Path, key: str) -> None:
    """Upload ``path`` to the weights bucket at ``key``, then check that its public URL
    serves it. Credentials and the endpoint come from the standard AWS variables."""
    try:
        client = boto3.client(
            "s3",
            config=Config(
                s3={"addressing_style": "path"}, request_checksum_calculation="when_required"
            ),
        )
        client.upload_file(
            str(path), WEIGHTS_BUCKET, key, ExtraArgs={"ContentType": "application/octet-stream"}
        )
    except (BotoCoreError, ClientError, S3UploadFailedError) as error:
        raise PublishError(
            f"Uploading {path.name} to s3://{WEIGHTS_BUCKET}/{key} failed: {error}. Set "
            "AWS_ENDPOINT_URL_S3, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY to the publisher "
            "key (docs/weights.md), or pass --no-checkpoint to publish without weights."
        ) from error
    url = f"{WEIGHTS_URL}/{key}"
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as r:
            served = int(r.headers.get("Content-Length", -1))
    except OSError as error:
        raise PublishError(
            f"Uploaded {path.name}, but {url} is not publicly readable: {error}. The bucket "
            "needs anonymous read and the public route on diagnostics.harville.dev."
        ) from error
    if served != path.stat().st_size:
        raise PublishError(f"{url} serves {served} bytes, expected {path.stat().st_size}")


def _license_allows_sharing(run_dir: Path) -> bool:
    record = json.loads((run_dir / "run.json").read_text())
    return record.get("dataset", {}).get("license") in OPEN_LICENSES


def publish_run(
    run_dir: Path,
    project_dir: Path,
    include_predictions: bool = True,
    allow_dirty: bool = False,
    include_checkpoint: bool = True,
    upload: Upload = upload_weights,
) -> Path:
    """Copy a finished run into ``project_dir/runs/<run-id>``.

    The small files are committed as they are. When the dataset's license allows sharing
    trained weights and ``include_checkpoint`` is true, the checkpoint is uploaded to
    ``<project>/<run-id>/checkpoint.ckpt`` in the weights bucket. ``artifacts.yaml`` records
    its size and SHA-256 either way, and its URL once uploaded. Rerunning on the same run
    keeps the recorded URL and uploads nothing.
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
        if url is None and include_checkpoint and _license_allows_sharing(run_dir):
            key = f"{project_dir.name}/{run_dir.name}/{name}"
            upload(source, key)
            url = f"{WEIGHTS_URL}/{key}"
        artifacts.append(Artifact(file=name, bytes=source.stat().st_size, sha256=digest, url=url))
    (target / "artifacts.yaml").write_text(
        yaml.safe_dump(Artifacts(files=artifacts).model_dump(), sort_keys=False)
    )
    return target
