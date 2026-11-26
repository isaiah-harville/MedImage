from pathlib import Path

import pytest

from medimage.registry import load_builtins

ROOT = Path(__file__).resolve().parents[1]

load_builtins()


@pytest.fixture
def root() -> Path:
    return ROOT
