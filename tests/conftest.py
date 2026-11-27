import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(autouse=True, scope="session")
def _isolated_matplotlib(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    mp.setenv("MPLCONFIGDIR", str(tmp_path_factory.mktemp("matplotlib")))
    yield
    mp.undo()


def fit_smoke(name: str, out: Path, *overrides: str) -> Path:
    from opendiagnostics.cli import fit

    return fit(["-c", str(ROOT / f"configs/smoke/{name}.yaml"), f"--output_dir={out}", *overrides])


@pytest.fixture(scope="session")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A smoke classification run, marked as made on a clean tree."""
    run_dir = fit_smoke("classification", tmp_path_factory.mktemp("runs"))
    record = json.loads((run_dir / "run.json").read_text())
    record["git"] = {"commit": "0" * 40, "branch": "main", "dirty": False}
    (run_dir / "run.json").write_text(json.dumps(record))
    return run_dir


class MemoryBucket:
    """The weights bucket in memory: ``publish_run`` writes it, and ``open_url`` serves it to
    ``weights.sync`` the way the anonymous public route does."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.served: list[str] = []

    def put(self, path: Path, key: str) -> None:
        self.objects[key] = path.read_bytes()

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def open_url(self, url: str):
        import io
        import urllib.error

        self.served.append(url)
        key = url.split("/opendiagnostics-weights/", 1)[1]
        if key not in self.objects:
            raise urllib.error.HTTPError(url, 403, "Access Denied", None, None)  # ty: ignore[invalid-argument-type]
        return io.BytesIO(self.objects[key])


def smoke_project(finished_run: Path, models: Path, **publish: Any) -> Path:
    from opendiagnostics.publish import publish_run

    project = models / "smoke-project"
    project.mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/smoke/classification.yaml", project / "config.yaml")
    publish_run(finished_run, project, **publish)
    return project


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "smoke-project"
    path.mkdir(parents=True)
    (path / "config.yaml").write_text((ROOT / "configs/smoke/classification.yaml").read_text())
    return path
