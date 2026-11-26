"""Experiment configuration.

An experiment is one YAML file that maps onto :class:`ExperimentConfig`. Parsing is strict:
unknown keys and wrong types are errors, so a typo cannot silently fall back to a default.
Every run writes the fully resolved config next to its results.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator


class _Loader(yaml.SafeLoader):
    """YAML 1.1 reads ``1e-4`` as a string; this loader reads it as a float, as YAML 1.2 does."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _yaml_load(text: Any) -> Any:
    return yaml.load(text, Loader=_Loader)


class ConfigError(ValueError):
    """Raised when a config file does not match the schema."""


def _list_to_tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


Fractions = Annotated[tuple[float, float, float], BeforeValidator(_list_to_tuple)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Component(_Strict):
    """A registry entry to build: ``name`` plus keyword ``params`` for its builder.

    A bare string is shorthand for a component with no params. ``params`` stays untyped
    here because each builder's own signature validates it when the component is built.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_name(cls, value: object) -> object:
        return {"name": value} if isinstance(value, str) else value


class SplitConfig(_Strict):
    """How to divide the dataset into train, validation, and test sets.

    ``predefined`` uses the dataset's official split. ``patient`` draws a random split in which
    every patient lands in exactly one subset, stratified by label when ``stratify`` is true.
    """

    method: Literal["predefined", "patient"] = "predefined"
    fractions: Fractions = (0.7, 0.15, 0.15)
    stratify: bool = True
    seed: int | None = None


class DataConfig(_Strict):
    dataset: Component
    split: SplitConfig = Field(default_factory=SplitConfig)
    train_transforms: list[Component] = Field(default_factory=list)
    eval_transforms: list[Component] = Field(default_factory=list)
    batch_size: int = 32
    eval_batch_size: int | None = None
    num_workers: int = 0


class OptimConfig(_Strict):
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: Literal["none", "cosine", "step"] = "cosine"
    warmup_epochs: int = 0
    step_size: int = 10
    gamma: float = 0.1


class TrainConfig(_Strict):
    epochs: int = 10
    device: str = "auto"
    amp: bool = False
    grad_clip_norm: float | None = None
    monitor: str | None = None
    monitor_mode: Literal["max", "min"] = "max"
    early_stopping_patience: int | None = None
    log_every: int = 50
    max_steps_per_epoch: int | None = None


class EvalConfig(_Strict):
    """Test-set evaluation. Confidence intervals come from a percentile bootstrap.

    ``bootstrap_unit: patient`` resamples whole patients (a cluster bootstrap), which keeps
    correlated images of one patient together and gives honest intervals.
    """

    bootstrap_samples: int = 1000
    confidence: float = 0.95
    bootstrap_unit: Literal["patient", "sample"] = "patient"
    splits: list[str] = Field(default_factory=lambda: ["val", "test"])


class ExperimentConfig(_Strict):
    name: str
    task: Component
    data: DataConfig
    model: Component
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    seed: int = 0
    deterministic: Literal["strict", "warn", "off"] = "strict"
    output_dir: str = "runs"
    plugins: list[str] = Field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def parse_config(data: object, where: str = "config") -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as error:
        problems = "\n".join(
            f"  {'.'.join(map(str, e['loc'])) or '(top level)'}: {e['msg']}" for e in error.errors()
        )
        raise ConfigError(f"{where} is not a valid experiment config:\n{problems}") from None


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides. Values are parsed as YAML (``2``, ``true``, ``[1, 2]``)."""
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep or not key:
            raise ConfigError(f"override {override!r} must look like key.path=value")
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"override {override!r}: {part!r} is not a mapping")
            node = child
        node[parts[-1]] = _yaml_load(raw)
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open() as handle:
        data = _yaml_load(handle) or {}
    return parse_config(apply_overrides(data, overrides or []), where=path.name)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
