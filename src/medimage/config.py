"""Experiment configuration.

An experiment is one YAML file that maps onto :class:`ExperimentConfig`. Parsing is strict:
unknown keys and wrong types are errors, so a typo cannot silently fall back to a default.
Every run writes the fully resolved config next to its results.
"""

from __future__ import annotations

import dataclasses
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


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


@dataclass
class Component:
    """A registry entry to build: ``name`` plus keyword ``params`` for its builder."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    """How to divide the dataset into train, validation, and test sets.

    ``predefined`` uses the dataset's official split. ``patient`` draws a random split in which
    every patient lands in exactly one subset, stratified by label when ``stratify`` is true.
    """

    method: Literal["predefined", "patient"] = "predefined"
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    stratify: bool = True
    seed: int | None = None


@dataclass
class DataConfig:
    dataset: Component
    split: SplitConfig = field(default_factory=SplitConfig)
    train_transforms: list[Component] = field(default_factory=list)
    eval_transforms: list[Component] = field(default_factory=list)
    batch_size: int = 32
    eval_batch_size: int | None = None
    num_workers: int = 0


@dataclass
class OptimConfig:
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: Literal["none", "cosine", "step"] = "cosine"
    warmup_epochs: int = 0
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 10
    device: str = "auto"
    amp: bool = False
    grad_clip_norm: float | None = None
    monitor: str | None = None
    monitor_mode: Literal["max", "min"] = "max"
    early_stopping_patience: int | None = None
    log_every: int = 50
    max_steps_per_epoch: int | None = None


@dataclass
class EvalConfig:
    """Test-set evaluation. Confidence intervals come from a percentile bootstrap.

    ``bootstrap_unit: patient`` resamples whole patients (a cluster bootstrap), which keeps
    correlated images of one patient together and gives honest intervals.
    """

    bootstrap_samples: int = 1000
    confidence: float = 0.95
    bootstrap_unit: Literal["patient", "sample"] = "patient"
    splits: list[str] = field(default_factory=lambda: ["val", "test"])


@dataclass
class ExperimentConfig:
    name: str
    task: Component
    data: DataConfig
    model: Component
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    deterministic: Literal["strict", "warn", "off"] = "strict"
    output_dir: str = "runs"
    plugins: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(v) for v in value]
    return value


def from_dict(cls: type, data: Any, where: str = "config") -> Any:
    """Build dataclass ``cls`` from plain data, checking keys and types."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed keys are {sorted(fields)}"
        )
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], f"{where}.{name}")
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{where}: missing required key {name!r}")
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, where)
            except ConfigError as error:
                errors.append(str(error))
        raise ConfigError("; ".join(errors))
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: expected one of {list(args)}, got {value!r}")
        return value
    if dataclasses.is_dataclass(tp):
        if isinstance(tp, type) and issubclass(tp, Component) and isinstance(value, str):
            return Component(name=value)
        return from_dict(tp, value, where)  # type: ignore[arg-type]
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, list | tuple) or len(value) != len(args):
            raise ConfigError(f"{where}: expected a list of {len(args)} items, got {value!r}")
        return tuple(
            _coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return dict(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported type {tp!r}")


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
    data = apply_overrides(data, overrides or [])
    return from_dict(ExperimentConfig, data, where=path.name)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
