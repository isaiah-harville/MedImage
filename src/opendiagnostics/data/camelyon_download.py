"""Resumable, MD5-checked downloads from the public CAMELYON bucket on AWS open data.

The bucket ``s3://camelyon-dataset`` (https://registry.opendata.aws/camelyon/) needs no
account and serves each slide as its own ``.tif``, so there is nothing to unpack. A file
downloads to ``<name>.partial``, resumes from the bytes already there, and is renamed only
after its MD5 matches the dataset's ``checksums.md5``. The CAMELYON data modules call
:func:`fetch_file` one slide at a time; ``scripts/datasets/`` call :func:`download` to fetch
whole folders and write ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "camelyon-dataset"
CHUNK = 16 * 1024 * 1024
CHECKSUMS = "checksums.md5"
METADATA = ("README.md", "license.txt", CHECKSUMS)


def client() -> Any:
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2")


def list_sizes(s3: Any, prefix: str) -> dict[str, int]:
    """Size in bytes of every object under ``prefix``, keyed by path relative to it."""
    sizes = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            relative = item["Key"].removeprefix(prefix)
            if relative and not relative.endswith("/"):
                sizes[relative] = item["Size"]
    return sizes


def read_checksums(path: Path) -> dict[str, str]:
    """``md5sum`` output (``<md5> *<path>``) as ``{path: md5}``."""
    sums = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.lstrip("*")] = digest
    return sums


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(
    s3: Any, dataset: str, name: str, target: Path, expected_md5: str | None = None
) -> str:
    """Download ``<dataset>/<name>`` to ``target`` and return its MD5.

    The bytes go to ``target.partial`` first, resuming from what is already there. A file
    whose MD5 differs from ``expected_md5`` is deleted and reported.
    """
    key = f"{dataset}/{name}"
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    partial = target.with_name(target.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    done = partial.stat().st_size if partial.exists() else 0
    if done > size:
        partial.unlink()
        done = 0
    if done < size:
        body = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={done}-")["Body"]
        with partial.open("ab") as handle:
            for block in body.iter_chunks(CHUNK):
                handle.write(block)
    if partial.stat().st_size != size:
        raise RuntimeError(f"{key}: got {partial.stat().st_size} bytes, expected {size}; rerun")
    actual = md5(partial)
    if expected_md5 and actual != expected_md5:
        partial.unlink()
        raise RuntimeError(f"{key}: MD5 {actual} does not match {expected_md5}; rerun to fetch it")
    partial.replace(target)
    return actual


def fetch_metadata(s3: Any, dataset: str, out: Path, names: tuple[str, ...]) -> None:
    """Fetch the small label and checksum files that are not in ``out`` yet."""
    for name in names:
        if not (out / name).exists():
            fetch_file(s3, dataset, name, out / name)


def download(
    dataset: str, out: Path, wanted: Callable[[str], bool], dry_run: bool = False
) -> dict[str, Any]:
    """Download the metadata files and every file whose relative path ``wanted`` accepts."""
    s3 = client()
    sizes = list_sizes(s3, f"{dataset}/")
    selected = sorted(name for name in sizes if wanted(name) and name not in METADATA)
    total = sum(sizes[name] for name in selected)
    print(f"{dataset}: {len(selected)} files, {total / 1e9:.1f} GB, into {out}")
    if dry_run:
        return {"files": len(selected), "bytes": total}
    fetch_metadata(s3, dataset, out, METADATA)
    checksums = read_checksums(out / CHECKSUMS)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for n, name in enumerate(selected, start=1):
        expected = checksums.get(name)
        entry = manifest.get(name, {})
        if (out / name).exists() and entry.get("verified") and entry.get("md5") == expected:
            continue
        actual = md5(out / name) if (out / name).exists() else None
        if actual is None or actual != expected:
            actual = fetch_file(s3, dataset, name, out / name, expected)
        manifest[name] = {"bytes": sizes[name], "md5": actual, "verified": expected is not None}
        partial_manifest = manifest_path.with_suffix(".partial")
        partial_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        partial_manifest.replace(manifest_path)
        print(f"{n}/{len(selected)} {name}")
    return {"files": len(selected), "bytes": total}


def script_parser(dataset: str, description: str) -> argparse.ArgumentParser:
    """The command line shared by ``scripts/datasets/camelyon16.py`` and ``camelyon17.py``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, default=Path("data") / dataset.lower())
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["images"],
        choices=["images", "annotations", "masks"],
        help="folders to download besides the label and checksum files (default: images)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes only")
    return parser
