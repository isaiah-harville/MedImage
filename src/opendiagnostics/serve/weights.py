"""Fetch the bundled models' checkpoints into a cache: ``opendiagnostics-fetch-weights``.

The image carries each model's run files but not its checkpoint. This downloads each
checkpoint from the weights bucket into ``WEIGHTS_CACHE/<sha256>.ckpt``, checking its size and
SHA-256 before it takes that name, so a checkpoint already in the cache is never downloaded
again. Checkpoints no bundled model uses are deleted. The service then loads from the cache.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from opendiagnostics.publish import WEIGHTS_URL, Artifact, load_artifacts

CHECKPOINT = "checkpoint.ckpt"
# Cloudflare, in front of the public weights URL, refuses Python's default User-Agent.
USER_AGENT = "opendiagnostics-serve"

log = logging.getLogger("opendiagnostics.serve")

Open = Callable[[str], BinaryIO]


def _open(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def checkpoint_of(folder: Path) -> Artifact | None:
    for artifact in load_artifacts(folder / "artifacts.yaml").files:
        if artifact.file == CHECKPOINT and artifact.url:
            return artifact
    return None


def cached(cache: Path, artifact: Artifact) -> Path:
    return cache / f"{artifact.sha256}.ckpt"


def _download(url: str, target: Path, artifact: Artifact, open_url: Open) -> None:
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


def fetch(weights_dir: Path, cache: Path, base_url: str, open_url: Open = _open) -> list[str]:
    """Download every bundled model's checkpoint missing from ``cache``. Returns their names."""
    cache.mkdir(parents=True, exist_ok=True)
    wanted, fetched = set(), []
    for folder in sorted(p for p in weights_dir.iterdir() if p.is_dir()):
        artifact = checkpoint_of(folder)
        if artifact is None:
            continue
        assert artifact.url
        if not artifact.url.startswith(f"{WEIGHTS_URL}/"):
            raise ValueError(
                f"{folder.name}: checkpoint URL {artifact.url} is not in {WEIGHTS_URL}"
            )
        target = cached(cache, artifact)
        wanted.add(target.name)
        if target.is_file():
            continue
        url = f"{base_url.rstrip('/')}/{artifact.url.removeprefix(f'{WEIGHTS_URL}/')}"
        log.info("downloading %s", url)
        _download(url, target, artifact, open_url)
        fetched.append(folder.name)
    for stale in cache.iterdir():
        if stale.name not in wanted:
            stale.unlink()
    return fetched


def main() -> None:
    from opendiagnostics.serve import settings_from

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = settings_from(dict(os.environ))
    fetched = fetch(settings.weights_dir, settings.weights_cache, str(settings.weights_base_url))
    log.info("downloaded %d checkpoints into %s", len(fetched), settings.weights_cache)
