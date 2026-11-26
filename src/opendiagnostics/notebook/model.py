"""What the lab notebook is built from: model projects, their published runs and baselines.

These are read from ``models/<project>/``. Files read here cross a boundary (they are hand
edited or written by earlier runs), so the YAML ones are parsed with strict schemas.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from opendiagnostics.publish import Artifacts, load_artifacts


class Baseline(BaseModel):
    """A published result on the same benchmark, quoted from its source."""

    model_config = ConfigDict(extra="forbid", strict=True)

    method: str
    metrics: dict[str, float]
    source: str
    url: str
    notes: str = ""


class Baselines(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    benchmark: str
    protocol: str
    baselines: list[Baseline] = Field(default_factory=list)


@dataclass(frozen=True)
class Run:
    project: str
    run_id: str
    path: Path
    config: dict[str, Any]
    record: dict[str, Any]
    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    splits: dict[str, Any]
    artifacts: Artifacts

    @property
    def monitor(self) -> str:
        return self.metrics["selection"]["monitor"].removeprefix("val/")

    def headline(self, split: str = "test") -> dict[str, Any] | None:
        value = self.metrics.get(split, {}).get(self.monitor)
        return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class Project:
    name: str
    path: Path
    card: str
    config: dict[str, Any]
    baselines: Baselines | None
    runs: list[Run] = field(default_factory=list)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_history(path: Path) -> list[dict[str, Any]]:
    """Lightning's CSVLogger rows, merged into one entry per epoch."""
    epochs: dict[int, dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            entry = epochs.setdefault(int(row["epoch"]), {"epoch": int(row["epoch"]), "val": {}})
            for key, value in row.items():
                if value in ("", None) or key in ("epoch", "step"):
                    continue
                if key == "train/loss":
                    entry["train_loss"] = float(value)
                elif key.startswith("val/"):
                    entry["val"][key.removeprefix("val/")] = float(value)
    return [epochs[e] for e in sorted(epochs)]


def load_run(project: str, path: Path) -> Run:
    history = read_history(path / "metrics.csv")
    splits = _json(path / "splits.json")
    return Run(
        project=project,
        run_id=path.name,
        path=path,
        config=yaml.safe_load((path / "config.yaml").read_text()),
        record=_json(path / "run.json"),
        metrics=_json(path / "metrics.json"),
        history=history,
        splits={k: v for k, v in splits.items() if k != "ids"},
        artifacts=load_artifacts(path / "artifacts.yaml"),
    )


def load_project(path: Path) -> Project:
    card_path = path / "MODEL_CARD.md"
    baselines_path = path / "baselines.yaml"
    runs_dir = path / "runs"
    run_paths = sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else []
    return Project(
        name=path.name,
        path=path,
        card=card_path.read_text() if card_path.exists() else "",
        config=yaml.safe_load((path / "config.yaml").read_text()),
        baselines=(
            Baselines.model_validate(yaml.safe_load(baselines_path.read_text()))
            if baselines_path.exists()
            else None
        ),
        runs=[load_run(path.name, p) for p in reversed(run_paths)],
    )


def load_projects(models_dir: Path) -> list[Project]:
    """Every directory in ``models_dir`` with a config.yaml, except templates (``_name``)."""
    return [
        load_project(p)
        for p in sorted(models_dir.iterdir())
        if p.is_dir() and not p.name.startswith("_") and (p / "config.yaml").exists()
    ]
