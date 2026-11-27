"""Helpers the dataset scripts share: resumable downloads, extraction, and manifest writing."""

from __future__ import annotations

import csv
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _download import retrying

Opener = Callable[..., Any]
DONE = ".complete"


@retrying
def download(url: str, dest: Path, opener: Opener = urllib.request.urlopen) -> Path:
    """Download ``url`` to ``dest``, resuming ``dest.part`` with a Range request after an
    interruption. A finished ``dest`` is never fetched again."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "opendiagnostics-dataset-script"}
    if done:
        headers["Range"] = f"bytes={done}-"
    try:
        response = opener(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as error:
        if error.code != 416:
            raise
        part.replace(dest)
        return dest
    with response:
        if done and response.status != 206:
            done = 0
        length = response.headers.get("Content-Length")
        total = done + int(length) if length else None
        print(f"downloading {url}" + (f" ({total / 1e9:.2f} GB)" if total else ""), flush=True)
        with part.open("ab" if done else "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    size = part.stat().st_size
    if total is not None and size != total:
        raise OSError(f"{url} stopped at {size} of {total} bytes; rerun to resume")
    part.replace(dest)
    return dest


def extract(archive: Path, dest: Path, flatten: bool = False) -> Path:
    """Unzip ``archive`` into ``dest`` once; a marker file records that it finished.

    ``flatten`` writes every file straight into ``dest`` under its own name, dropping the
    archive's folders. Deeply nested archives otherwise pass Windows' 260-character path
    limit. Two members with the same name are an error.
    """
    if (dest / DONE).exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        if not flatten:
            handle.extractall(dest)
        else:
            seen: set[str] = set()
            for member in handle.infolist():
                name = Path(member.filename).name
                if member.is_dir() or not name:
                    continue
                if name in seen:
                    raise RuntimeError(f"{archive} holds two files named {name}; cannot flatten")
                seen.add(name)
                with handle.open(member) as src, (dest / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
    (dest / DONE).touch()
    return dest


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
