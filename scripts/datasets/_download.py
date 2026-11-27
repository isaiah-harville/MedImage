"""Resumable downloads, safe extraction, and manifest writing for the dataset scripts here."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

USER_AGENT = "opendiagnostics"
CHUNK = 8 * 1024 * 1024
REPORT_EVERY = 512 * 1024 * 1024
ATTEMPTS = 5
MANIFEST_COLUMNS = ("path", "patient_id", "label", "split")
Opener = Callable[[urllib.request.Request], Any]


class DownloadError(RuntimeError):
    """Raised when a download or an archive does not match what its source publishes."""


def parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("out", type=Path, help="dataset directory, such as data/<name>")
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="delete each downloaded archive once it is extracted",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _resume(url: str, partial: Path, size: int, opener: Opener) -> None:
    done = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if done:
        headers["Range"] = f"bytes={done}-"
    with opener(urllib.request.Request(url, headers=headers)) as response:
        if done and getattr(response, "status", 200) != 206:
            done = 0
        with partial.open("ab" if done else "wb") as out:
            reported = done // REPORT_EVERY
            while block := response.read(CHUNK):
                out.write(block)
                done += len(block)
                if done // REPORT_EVERY > reported:
                    reported = done // REPORT_EVERY
                    print(f"  {partial.name}: {done / 2**30:.1f} of {size / 2**30:.1f} GiB")


def fetch(
    url: str,
    dest: Path,
    size: int,
    sha256: str | None = None,
    opener: Opener = urllib.request.urlopen,
) -> Path:
    """Download ``url`` to ``dest``, resuming a ``.part`` file left by an earlier run.

    The file is renamed into place only once its size, and its SHA-256 when given, match.
    """
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    print(f"downloading {dest.name} ({size / 2**30:.2f} GiB)")
    for attempt in range(1, ATTEMPTS + 1):
        if partial.exists() and partial.stat().st_size == size:
            break
        try:
            _resume(url, partial, size, opener)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            if attempt == ATTEMPTS:
                raise DownloadError(
                    f"{dest.name} failed after {ATTEMPTS} attempts ({error}). Rerun the script "
                    "to resume from the bytes already saved."
                ) from error
            print(f"  {dest.name}: {error}; resuming")
    got = partial.stat().st_size if partial.exists() else 0
    if got != size:
        raise DownloadError(
            f"{dest.name} is {got} bytes, but its source lists {size}. Rerun to resume; if it "
            "persists, the source has changed and the size in the script needs updating."
        )
    if sha256 and (actual := _sha256(partial)) != sha256:
        partial.unlink()
        raise DownloadError(
            f"{dest.name} has SHA-256 {actual}, but its source publishes {sha256}. The partial "
            "file was deleted; rerun to download it again."
        )
    partial.replace(dest)
    return dest


def _marker(out: Path, archive: Path) -> Path:
    return out / f".{archive.name}.extracted"


def extract(archive: Path, out: Path, rename: Callable[[str], str | None] | None = None) -> None:
    """Unpack a tar or zip archive into ``out``; ``rename`` maps or drops zip member names.

    Tar members are filtered with the standard library's ``data`` filter, and zip members
    must stay inside ``out``. A marker file records a finished extraction.
    """
    out.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name}")
    if zipfile.is_zipfile(archive):
        root = out.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                name = rename(info.filename) if rename else info.filename
                if name is None or info.is_dir():
                    continue
                target = (out / name).resolve()
                if not target.is_relative_to(root):
                    raise DownloadError(f"{archive.name} member {info.filename!r} leaves {out}")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                with bundle.open(info) as src, partial.open("wb") as dst:
                    while block := src.read(CHUNK):
                        dst.write(block)
                partial.replace(target)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(out, filter="data")
    _marker(out, archive).touch()


def fetch_and_extract(
    url: str,
    archive: Path,
    size: int,
    out: Path,
    delete: bool,
    sha256: str | None = None,
    rename: Callable[[str], str | None] | None = None,
    opener: Opener = urllib.request.urlopen,
) -> None:
    """Download and unpack ``archive`` unless an earlier run already unpacked it into ``out``."""
    if not _marker(out, archive).exists():
        fetch(url, archive, size, sha256, opener)
        extract(archive, out, rename)
    if delete and archive.exists():
        archive.unlink()
        print(f"deleted {archive.name}")


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> int:
    """Write ``rows`` sorted by path to ``path`` atomically; returns the row count."""
    rows = sorted(rows, key=lambda row: row["path"])
    partial = path.with_name(path.name + ".part")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)
    return len(rows)
