import pytest
from xpt import expect

from opendiagnostics.config import (
    ConfigError,
    apply_overrides,
    dump_config,
    load_config,
    parse_config,
)

MINIMAL = {
    "name": "x",
    "task": "classification",
    "data": {"dataset": {"name": "synthetic_classification"}},
    "model": {"name": "simple_cnn", "params": {"width": 4}},
}


def test_minimal_config_fills_defaults():
    config = parse_config(MINIMAL)
    assert config.task.name == "classification"
    assert config.data.split.method == "predefined"
    assert config.eval.bootstrap_unit == "patient"


def test_unknown_keys_and_wrong_types_are_reported_with_their_paths():
    data = {**MINIMAL, "train": {"epoch": 3, "epochs": "3"}, "deterministic": "yes"}
    with pytest.raises(ConfigError) as error:
        parse_config(data, "x.yaml")
    expect(
        str(error.value),
        """x.yaml is not a valid experiment config:
  train.epochs: Input should be a valid integer
  train.epoch: Extra inputs are not permitted
  deterministic: Input should be 'strict', 'warn' or 'off'""",
    )


def test_overrides_parse_yaml_values():
    data = apply_overrides(
        {"train": {"epochs": 1}}, ["train.epochs=5", "optim.lr=1e-4", "train.amp=true"]
    )
    assert data == {"train": {"epochs": 5, "amp": True}, "optim": {"lr": 1e-4}}


def test_round_trip(tmp_path, root):
    config = load_config(root / "configs/smoke/classification.yaml")
    dump_config(config, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == config


def test_every_shipped_config_parses(root):
    paths = [*root.glob("configs/**/*.yaml"), *root.glob("models/*/config.yaml")]
    assert paths
    for path in paths:
        load_config(path)


def test_exponent_floats_load_as_numbers(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "name: x\ntask: classification\ndata: {dataset: synthetic_classification}\n"
        "model: simple_cnn\noptim: {lr: 1e-4, weight_decay: 5E-5}\n"
    )
    config = load_config(path)
    assert config.optim.lr == 1e-4
    assert config.optim.weight_decay == 5e-5
