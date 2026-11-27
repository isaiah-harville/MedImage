"""What a run needs to be audited or repeated: git state and the software environment."""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("opendiagnostics")

TRACKED_PACKAGES = (
    "opendiagnostics",
    "torch",
    "torchvision",
    "lightning",
    "torchmetrics",
    "timm",
    "monai",
    "medmnist",
    "numpy",
)


RUN_OUTPUTS = ":(exclude,glob)models/*/runs/**"


def git_info(path: Path | str = ".") -> dict[str, Any]:
    """Commit, branch, and whether tracked files had uncommitted changes.

    Published run outputs under ``models/*/runs/`` are not inputs to training, so changes to
    them (a republish in progress, say) do not make a run dirty.
    """

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=path, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None}
    dirty = bool(run("status", "--porcelain", "--untracked-files=no", "--", ".", RUN_OUTPUTS))
    if dirty:
        log.warning("uncommitted changes: this run cannot be reproduced from the commit alone")
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty}


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "mps_available": torch.backends.mps.is_available(),
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
