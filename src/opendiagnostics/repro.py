"""Seeding, deterministic execution, and run provenance."""

from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("opendiagnostics")

TRACKED_PACKAGES = (
    "opendiagnostics",
    "torch",
    "torchvision",
    "timm",
    "monai",
    "numpy",
    "pydicom",
    "nibabel",
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch on every device."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG that other libraries use
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators


def set_deterministic(mode: str) -> None:
    """``strict`` errors on nondeterministic ops, ``warn`` only warns, ``off`` allows them.

    CUDA needs ``CUBLAS_WORKSPACE_CONFIG`` set before the first cuBLAS call; it is set here
    if absent. Some ops (several CUDA backward passes used by detectors) have no deterministic
    implementation, so detection on GPU may need ``warn``.
    """
    if mode not in ("strict", "warn", "off"):
        raise ValueError(f"deterministic must be strict, warn, or off, got {mode!r}")
    enabled = mode != "off"
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled, warn_only=mode == "warn")
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and ``random`` in each DataLoader worker from torch's per-worker seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG that other libraries use
    random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def git_info(path: Path | str = ".") -> dict[str, Any]:
    """Commit, branch, and whether the working tree had uncommitted changes."""

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
    status = run("status", "--porcelain", "--untracked-files=no")
    info = {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }
    if info["dirty"]:
        log.warning("uncommitted changes: this run cannot be reproduced from the commit alone")
    return info


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "mps_available": torch.backends.mps.is_available(),
        "torch_threads": torch.get_num_threads(),
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
