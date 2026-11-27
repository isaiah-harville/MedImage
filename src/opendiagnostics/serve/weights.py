"""Download the published models from the weights bucket into a local cache.

``catalog.json`` in the bucket names each project's served run, with its bundle's SHA-256.
For each one, ``sync`` downloads ``<project>/<run-id>/demo.tar``, again when the hash
changes, (config, metrics, artifacts, examples) into
``WEIGHTS_CACHE/models/<project>/<run-id>/`` and the checkpoint into
``WEIGHTS_CACHE/checkpoints/<sha256>.ckpt``. A file takes its final name only after its size
and SHA-256 match ``artifacts.yaml``, so anything already in the cache is never downloaded
again. ``prune`` deletes what the catalog no longer names.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict

from opendiagnostics.publish import CATALOG, DEMO_BUNDLE, WEIGHTS_URL, Artifact, load_artifacts

CHECKPOINT = "checkpoint.ckpt"
COMPLETE = ".complete"
# Cloudflare, in front of the public weights URL, refuses Python's default User-Agent.
USER_AGENT = "opendiagnostics-serve"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

log = logging.getLogger("opendiagnostics.serve")

Open = Callable[[str], BinaryIO]


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, str]
    bundles: dict[str, str] = {}


def http_open(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def _retrying(open_url: Open, sleep: Callable[[float], None], tries: int = 6) -> Open:
    """Wait and retry when the public route's rate limit answers 429."""

    def opened(url: str) -> BinaryIO:
        for attempt in range(tries):
            try:
                return open_url(url)
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == tries - 1:
                    raise
                retry_after = error.headers.get("Retry-After") if error.headers else None
                sleep(float(retry_after or 10))
        raise AssertionError("unreachable")

    return opened


def checkpoint_of(folder: Path) -> Artifact | None:
    for artifact in load_artifacts(folder / "artifacts.yaml").files:
        if artifact.file == CHECKPOINT and artifact.url:
            return artifact
    return None


def cached(cache: Path, artifact: Artifact) -> Path:
    return cache / "checkpoints" / f"{artifact.sha256}.ckpt"


def read_catalog(base_url: str, open_url: Open) -> Catalog:
    try:
        with open_url(f"{base_url}/{CATALOG}") as response:
            catalog = Catalog.model_validate_json(response.read())
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            log.warning("no %s in the bucket (HTTP %d); serving no models", CATALOG, error.code)
            return Catalog(models={})
        raise
    for project, run in catalog.models.items():
        if not (NAME.match(project) and NAME.match(run)):
            raise ValueError(f"{CATALOG} names {project!r}: {run!r}, not a safe folder name")
    return catalog


def _download_checkpoint(url: str, target: Path, artifact: Artifact, open_url: Open) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    digest = hashlib.sha256()
    with open_url(url) as response, partial.open("wb") as handle:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            handle.write(chunk)
    size = partial.stat().st_size
    if size != artifact.bytes or digest.hexdigest() != artifact.sha256:
        partial.unlink()
        raise ValueError(
            f"{url} served {size} bytes with SHA-256 {digest.hexdigest()}; artifacts.yaml "
            f"expects {artifact.bytes} bytes with {artifact.sha256}"
        )
    os.replace(partial, target)


def _download_run(url: str, folder: Path, open_url: Open, bundle_sha: str) -> None:
    partial = folder.with_name(folder.name + ".part")
    bundle = folder.with_name(folder.name + ".tar")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    with open_url(url) as response, bundle.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    with tarfile.open(bundle) as tar:
        tar.extractall(partial, filter="data")
    bundle.unlink()
    (partial / COMPLETE).write_text(bundle_sha)
    shutil.rmtree(folder, ignore_errors=True)
    os.replace(partial, folder)


def _checkpoint_url(folder: Path, base_url: str) -> tuple[Artifact, str]:
    artifact = checkpoint_of(folder)
    url = artifact.url if artifact else None
    if artifact is None or url is None or not url.startswith(f"{WEIGHTS_URL}/"):
        raise ValueError(f"{folder}/artifacts.yaml has no checkpoint URL in {WEIGHTS_URL}")
    return artifact, f"{base_url}/{url.removeprefix(f'{WEIGHTS_URL}/')}"


def sync(
    cache: Path,
    base_url: str,
    open_url: Open = http_open,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Download every cataloged model missing from ``cache``. Returns the model folders.

    A model that fails to download is logged and left out, so one bad upload never stops
    the others from being served.
    """
    opened = _retrying(open_url, sleep)
    base_url = base_url.rstrip("/")
    folders = []
    catalog = read_catalog(base_url, opened)
    for project, run in sorted(catalog.models.items()):
        folder = cache / "models" / project / run
        bundle_sha = catalog.bundles.get(project, "")
        marker = folder / COMPLETE
        try:
            if not marker.exists() or marker.read_text() != bundle_sha:
                log.info("downloading %s/%s", project, run)
                url = f"{base_url}/{project}/{run}/{DEMO_BUNDLE}"
                _download_run(url, folder, opened, bundle_sha)
            artifact, url = _checkpoint_url(folder, base_url)
            target = cached(cache, artifact)
            if not target.is_file():
                log.info("downloading %s", url)
                _download_checkpoint(url, target, artifact, opened)
        except (OSError, ValueError, tarfile.TarError) as error:
            log.error("skipping %s/%s: %s", project, run, error)
            continue
        folders.append(folder)
    return folders


def prune(cache: Path, folders: list[Path]) -> None:
    """Delete cached models and checkpoints that ``folders`` do not use."""
    keep = set(folders)
    checkpoints = {cached(cache, a).name for f in folders if (a := checkpoint_of(f))}
    for project in (cache / "models").glob("*"):
        for run in project.iterdir():
            if run not in keep:
                if run.is_dir():
                    shutil.rmtree(run)
                else:
                    run.unlink()
        if not any(project.iterdir()):
            project.rmdir()
    for checkpoint in (cache / "checkpoints").glob("*"):
        if checkpoint.name not in checkpoints:
            checkpoint.unlink()
