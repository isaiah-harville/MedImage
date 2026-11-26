import tomllib

import opendiagnostics


def test_version_matches_pyproject(root):
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert opendiagnostics.__version__ == pyproject["project"]["version"]
