import importlib.util

import pytest

from opendiagnostics.cli import OpenDiagnosticsCLI


class ParseOnly(OpenDiagnosticsCLI):
    def instantiate_classes(self) -> None:
        """Validate types and keys without building networks or downloading data."""


def _configs(root):
    return [*root.glob("configs/smoke/*.yaml"), *root.glob("models/*/config.yaml")]


def test_every_config_parses_strictly(root):
    configs = [p for p in _configs(root) if "_template" not in p.parts]
    assert configs
    for path in configs:
        ParseOnly(args=["--config", str(path)], run=False)


def test_template_config_parses_once_filled_in(root, tmp_path):
    text = (root / "models/_template/config.yaml").read_text()
    (tmp_path / "labels.csv").write_text("path,patient_id,label\n")
    filled = tmp_path / "config.yaml"
    filled.write_text(text.replace("/path/to/labels.csv", str(tmp_path / "labels.csv")))
    ParseOnly(args=["--config", str(filled)], run=False)


def test_typo_in_a_config_is_an_error(root, tmp_path, capsys):
    path = root / "models/pathmnist-convnext/config.yaml"
    (tmp_path / "typo.yaml").write_text(path.read_text().replace("batch_size:", "batch_sise:"))
    with pytest.raises(SystemExit):
        ParseOnly(args=["--config", str(tmp_path / "typo.yaml")], run=False)
    assert "batch_sise" in capsys.readouterr().err


def test_medmnist_project_configs_match_their_generator(root):
    spec = importlib.util.spec_from_file_location("recipes", root / "scripts/medmnist_recipes.py")
    assert spec
    assert spec.loader
    recipes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipes)
    for flag in [*recipes.RECIPES, *recipes.RECIPES_3D]:
        written = (root / f"models/{recipes.project(flag)}/config.yaml").read_text()
        assert written == recipes.render(flag), (
            f"models/{recipes.project(flag)}/config.yaml is stale; run "
            "`uv run python scripts/medmnist_recipes.py`"
        )
