"""Name-to-builder registries and plugin loading.

Every extensible part of the pipeline (datasets, models, tasks, transforms) is a
:class:`Registry`. A plugin is any module that registers builders when imported. Plugins are
found three ways:

* the built-in modules of this package, imported by :func:`load_builtins`;
* installed packages that declare an entry point in the ``medimage.plugins`` group;
* modules or ``.py`` files named in an experiment config's ``plugins`` list.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

ENTRY_POINT_GROUP = "medimage.plugins"


class RegistryError(KeyError):
    """Raised for unknown or duplicate registry names."""

    def __str__(self) -> str:  # KeyError quotes its message; keep it readable.
        return str(self.args[0])


class Registry(Generic[T]):
    """A mapping from names to callables that build objects of one kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._builders: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[F], F]:
        """Decorator that registers a class or function under ``name``."""

        def decorator(builder: F) -> F:
            existing = self._builders.get(name)
            if existing is not None and existing is not builder:
                raise RegistryError(f"{self.kind} {name!r} is already registered by {existing!r}")
            self._builders[name] = builder
            return builder

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        try:
            return self._builders[name]
        except KeyError:
            close = difflib.get_close_matches(name, self._builders, n=3)
            hint = f" Did you mean {', '.join(map(repr, close))}?" if close else ""
            raise RegistryError(
                f"unknown {self.kind} {name!r}.{hint} Registered: {', '.join(self.names())}"
            ) from None

    def build(self, name: str, **params: Any) -> T:
        return self.get(name)(**params)

    def names(self) -> list[str]:
        return sorted(self._builders)

    def __contains__(self, name: object) -> bool:
        return name in self._builders

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"


DATASETS: Registry[Any] = Registry("dataset")
MODELS: Registry[Any] = Registry("model")
TASKS: Registry[Any] = Registry("task")
TRANSFORMS: Registry[Any] = Registry("transform")

_BUILTIN_MODULES = (
    "medimage.datasets",
    "medimage.models",
    "medimage.tasks",
    "medimage.transforms",
)
_loaded_entry_points = False


def load_builtins() -> None:
    """Import the modules that register the built-in components."""
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)


def load_entry_points() -> None:
    """Import every installed plugin that declares a ``medimage.plugins`` entry point."""
    global _loaded_entry_points
    if _loaded_entry_points:
        return
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        entry_point.load()
    _loaded_entry_points = True


def load_plugin(spec: str, base_dir: Path | None = None) -> None:
    """Import a plugin given as a dotted module name or a path to a ``.py`` file.

    Relative paths resolve against ``base_dir``, normally the directory of the config file.
    Importing a plugin runs its code, so only list plugins you trust.
    """
    if not spec.endswith(".py"):
        importlib.import_module(spec)
        return
    path = Path(spec)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"plugin file not found: {path}")
    module_name = "medimage_plugin_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    if module_name in sys.modules:
        return
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import plugin {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)


def load_all(plugins: Iterable[str] = (), base_dir: Path | None = None) -> None:
    load_builtins()
    load_entry_points()
    for spec in plugins:
        load_plugin(spec, base_dir)
