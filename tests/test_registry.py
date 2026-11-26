import pytest

from opendiagnostics.registry import (
    DATASETS,
    MODELS,
    TASKS,
    TRANSFORMS,
    Registry,
    RegistryError,
    load_plugin,
)


def test_builtins_are_registered():
    assert {"medmnist", "csv_classification", "coco_detection", "synthetic_classification"} <= set(
        DATASETS.names()
    )
    assert {"simple_cnn", "timm", "monai", "fasterrcnn", "retinanet", "fcos"} <= set(MODELS.names())
    assert {"classification", "detection"} <= set(TASKS.names())
    assert {"resize", "random_flip", "normalize_intensity"} <= set(TRANSFORMS.names())


def test_unknown_name_suggests_close_match():
    with pytest.raises(RegistryError, match="Did you mean 'simple_cnn'"):
        MODELS.get("simple_cn")


def test_duplicate_registration_is_an_error():
    registry: Registry = Registry("thing")
    registry.register("a")(lambda: 1)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("a")(lambda: 2)


def test_file_plugin_registers_components(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        "from opendiagnostics.registry import MODELS\n"
        "@MODELS.register('test_plugin_model')\n"
        "def build(**kwargs):\n"
        "    return kwargs\n"
    )
    load_plugin("plugin.py", base_dir=tmp_path)
    assert MODELS.build("test_plugin_model", a=1) == {"a": 1}
