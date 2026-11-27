"""Resumable, MD5-checked downloads from the public CAMELYON bucket on AWS open data.

The bucket ``s3://camelyon-dataset`` (https://registry.opendata.aws/camelyon/) needs no
account. It serves each slide as its own ``.tif``, so there are no archives to unpack. Each
file downloads to ``<name>.partial``, resumes from the bytes already there, and is renamed
only after its MD5 matches the dataset's ``checksums.md5``. ``manifest.json`` in the output
directory lists every selected file with its size, MD5 and whether it is verified.
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
METADATA = ("README.md", "license.txt", "checksums.md5")


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


def fetch(s3: Any, key: str, target: Path, size: int) -> None:
    """Download ``key`` to ``target.partial``, resuming from what is already there."""
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


def download(
    dataset: str, out: Path, wanted: Callable[[str], bool], dry_run: bool = False
) -> dict[str, Any]:
    """Download the metadata files and every file whose relative path ``wanted`` accepts."""
    s3 = client()
    prefix = f"{dataset}/"
    sizes = list_sizes(s3, prefix)
    selected = sorted(name for name in sizes if wanted(name) and name not in METADATA)
    total = sum(sizes[name] for name in selected)
    print(f"{dataset}: {len(selected)} files, {total / 1e9:.1f} GB, into {out}")
    if dry_run:
        return {"files": len(selected), "bytes": total}
    out.mkdir(parents=True, exist_ok=True)
    for name in METADATA:
        fetch(s3, prefix + name, out / name, sizes[name])
        (out / (name + ".partial")).replace(out / name)
    checksums = read_checksums(out / "checksums.md5")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for n, name in enumerate(selected, start=1):
        target = out / name
        expected = checksums.get(name)
        entry = manifest.get(name, {})
        if target.exists() and entry.get("verified") and entry.get("md5") == expected:
            continue
        if not target.exists():
            fetch(s3, prefix + name, target, sizes[name])
            target.with_name(target.name + ".partial").replace(target)
        actual = md5(target)
        if expected and actual != expected:
            target.unlink()
            raise RuntimeError(f"{name}: MD5 {actual} does not match {expected}; rerun to fetch it")
        manifest[name] = {"bytes": sizes[name], "md5": actual, "verified": expected is not None}
        partial_manifest = manifest_path.with_suffix(".partial")
        partial_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        partial_manifest.replace(manifest_path)
        print(f"{n}/{len(selected)} {name}")
    return {"files": len(selected), "bytes": total}


def parser(dataset: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data") / dataset.lower(),
        help="where the dataset goes; on another disk, point the config's root at it too",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["images"],
        choices=["images", "annotations", "masks"],
        help="folders to download besides the label and checksum files (default: images)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes only")
    return parser
