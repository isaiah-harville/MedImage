import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from medimage.config import load_config
from medimage.registry import load_builtins

ROOT = Path(__file__).resolve().parents[1]

load_builtins()


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture(autouse=True, scope="session")
def _isolated_matplotlib(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    mp.setenv("MPLCONFIGDIR", str(tmp_path_factory.mktemp("matplotlib")))
    yield
    mp.undo()


@pytest.fixture(scope="session")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A smoke classification run, marked as made on a clean tree."""
    from medimage.engine import train

    out = tmp_path_factory.mktemp("runs")
    config = load_config(ROOT / "configs/smoke/classification.yaml", [f"output_dir={out}"])
    run_dir = train(config, ROOT)
    record = json.loads((run_dir / "run.json").read_text())
    record["git"] = {"commit": "0" * 40, "branch": "main", "dirty": False}
    (run_dir / "run.json").write_text(json.dumps(record))
    return run_dir


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "smoke-project"
    path.mkdir(parents=True)
    (path / "config.yaml").write_text((ROOT / "configs/smoke/classification.yaml").read_text())
    return path
