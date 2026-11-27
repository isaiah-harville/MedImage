import importlib.util
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
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


def bundler() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bundle", ROOT / "scripts/bundle_models.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_project(finished_run: Path, models: Path, **publish: Any) -> Path:
    from opendiagnostics.publish import publish_run

    project = models / "smoke-project"
    project.mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/smoke/classification.yaml", project / "config.yaml")
    publish_run(finished_run, project, **publish)
    return project


def bundle_published(finished_run: Path, tmp: Path) -> Path:
    """Publish ``finished_run`` into a smoke project and bundle it; returns the bundle."""
    smoke_project(finished_run, tmp / "models", upload=lambda path, key: None)
    weights = tmp / "weights"
    assert bundler().main([str(weights), "--models", str(tmp / "models")]) == 0
    return weights


def serve_checkpoint(path: Path, served: list[str] | None = None):
    """An opener for ``fetch`` that serves ``path`` for every URL and records the URLs."""

    def open_url(url: str):
        if served is not None:
            served.append(url)
        return path.open("rb")

    return open_url


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "smoke-project"
    path.mkdir(parents=True)
    (path / "config.yaml").write_text((ROOT / "configs/smoke/classification.yaml").read_text())
    return path
