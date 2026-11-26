import pytest

from medimage.config import (
    ConfigError,
    ExperimentConfig,
    apply_overrides,
    dump_config,
    from_dict,
    load_config,
)

MINIMAL = {
    "name": "x",
    "task": "classification",
    "data": {"dataset": {"name": "synthetic_classification"}},
    "model": {"name": "simple_cnn", "params": {"width": 4}},
}


def test_minimal_config_fills_defaults():
    config = from_dict(ExperimentConfig, MINIMAL)
    assert config.task.name == "classification"
    assert config.data.split.method == "predefined"
    assert config.eval.bootstrap_unit == "patient"


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['epoch'\]"):
        from_dict(ExperimentConfig, {**MINIMAL, "train": {"epoch": 3}})


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigError, match="expected an integer"):
        from_dict(ExperimentConfig, {**MINIMAL, "train": {"epochs": "3"}})
    with pytest.raises(ConfigError, match="expected one of"):
        from_dict(ExperimentConfig, {**MINIMAL, "deterministic": "yes"})


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
