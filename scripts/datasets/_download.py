"""Shared steps of the dataset download scripts: fetch, verify, extract, write the manifest.

Every step is idempotent. A finished download is kept only after its size (and MD5 or SHA-256,
when the source publishes one) matches; an interrupted one resumes from its ``.part`` file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

MANIFEST_COLUMNS = ("path", "patient_id", "label", "split")
CHUNK = 1 << 20
Opener = Callable[[urllib.request.Request], Any]


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("root", type=Path, help="dataset directory, for example data/isic2020")
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="delete each archive once it is extracted",
    )
    return parser.parse_args()


def md5sum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, size: int | None, md5: str | None, sha256: str | None = None) -> bool:
    if size is not None and path.stat().st_size != size:
        return False
    if sha256 is not None and md5sum(path, "sha256") != sha256:
        return False
    return md5 is None or md5sum(path) == md5


def download(
    url: str,
    dest: Path,
    size: int | None = None,
    md5: str | None = None,
    opener: Opener | None = None,
    sha256: str | None = None,
) -> Path:
    """Fetch ``url`` to ``dest``, resuming a partial download, and check its size and MD5 or
    SHA-256."""
    if dest.exists():
        if _matches(dest, size, md5, sha256):
            return dest
        raise RuntimeError(
            f"{dest} exists but does not match the published size {size} or MD5 {md5}; "
            "delete it and rerun to download it again"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have > size:
        part.unlink()
        have = 0
    if size is None or have < size:
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        print(f"downloading {url} -> {dest}" + (f" (resuming at {have} bytes)" if have else ""))
        with (opener or urllib.request.urlopen)(request) as response:
            resumed = have and getattr(response, "status", 200) == 206
            with part.open("ab" if resumed else "wb") as handle:
                _copy(response, handle)
    if not _matches(part, size, md5, sha256):
        got = part.stat().st_size
        part.unlink()
        raise RuntimeError(
            f"{url} downloaded {got} bytes that do not match the published size {size} or "
            f"MD5 {md5}; the partial file was deleted, so rerun to download it again"
        )
    part.replace(dest)
    return dest


def _copy(response: BinaryIO, handle: BinaryIO) -> None:
    while chunk := response.read(CHUNK):
        handle.write(chunk)


def extract(
    archive: Path,
    into: Path,
    delete_archive: bool = False,
    keep: Callable[[str], bool] | None = None,
) -> Path:
    """Unpack a zip or tar ``archive`` into ``into`` once; a marker file records that it
    finished. With ``keep``, only the members whose name it accepts are unpacked. Tar members
    that would land outside ``into`` are refused."""
    marker = into / f".{archive.name}.extracted"
    if not marker.exists():
        print(f"extracting {archive} -> {into}")
        into.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
                handle.extractall(into, [n for n in names if keep(n)] if keep else None)
        else:
            with tarfile.open(archive) as handle:
                members = [m for m in handle.getmembers() if keep is None or keep(m.name)]
                handle.extractall(into, members, filter="data")
        marker.touch()
    if delete_archive and archive.exists():
        archive.unlink()
    return into


def fetch_and_extract(
    url: str,
    archive: Path,
    into: Path,
    size: int | None = None,
    md5: str | None = None,
    delete_archive: bool = False,
    opener: Opener | None = None,
) -> Path:
    """Download and extract one archive, skipping the download if it was already extracted."""
    if not (into / f".{archive.name}.extracted").exists():
        download(url, archive, size, md5, opener)
    return extract(archive, into, delete_archive)


def find_files(root: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    """Map each file's stem to its path, for files under ``root`` with one of ``suffixes``."""
    wanted = {s.lower() for s in suffixes}
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in wanted and not path.name.startswith("."):
            if path.stem in found:
                raise RuntimeError(f"two files are named {path.stem}: {found[path.stem]}, {path}")
            found[path.stem] = path
    return found


def read_csv(path: Path, required: Iterable[str], delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    missing = set(required) - set(rows[0] if rows else ())
    if missing:
        raise RuntimeError(
            f"{path} lacks the columns {sorted(missing)}; the source may have changed its format"
        )
    return rows


def write_manifest(
    root: Path, rows: Iterable[Mapping[str, str]], columns: Iterable[str] = MANIFEST_COLUMNS
) -> Path:
    """Write ``manifest.csv`` atomically, with paths relative to ``root``."""
    rows = list(rows)
    out = root / "manifest.csv"
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)
    print(f"wrote {out}: {len(rows)} images")
    return out


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def assign_splits(
    groups: list[str],
    labels: list[str],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> list[str]:
    """Name each sample's split from a group-level split stratified by label.

    This is ``patient_split``, the same split ``split: patient`` makes, frozen into the manifest.
    """
    from opendiagnostics.data import patient_split

    names = [""] * len(groups)
    for split, indices in patient_split(groups, fractions, labels, seed).items():
        for index in indices:
            names[index] = split
    return names
