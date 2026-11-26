import tomllib

import medimage


def test_version_matches_pyproject(root):
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert medimage.__version__ == pyproject["project"]["version"]


def test_core_dependencies_exclude_monai(root):
    # MONAI is an optional extra; see docs/design.md.
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert not any(dep.startswith("monai") for dep in pyproject["project"]["dependencies"])
