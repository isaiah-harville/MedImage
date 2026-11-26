"""The one training and evaluation loop used by every task.

A run directory holds everything needed to audit or repeat an experiment::

    runs/<name>/<timestamp>-<commit>/
        config.yaml          the fully resolved config
        run.json             seed, git commit and dirty flag, package versions, hardware,
                             data fingerprint, command line, start and end times
        splits.json          sample IDs in each split, with patient and label counts
        history.jsonl        one line per epoch: training loss, validation metrics, lr
        metrics.json         final metrics with bootstrap confidence intervals
        predictions_<split>.csv
        checkpoint.pt        weights of the epoch with the best validation monitor
        train.log
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from medimage import repro
from medimage.config import ExperimentConfig, dump_config, load_config
from medimage.data.base import DatasetSource, SplitDataset
from medimage.data.splits import SPLIT_NAMES, check_disjoint, patient_split, summarize
from medimage.registry import DATASETS, MODELS, TASKS, load_all
from medimage.tasks import Task
from medimage.transforms import build_transforms

log = logging.getLogger("medimage")


@dataclass
class Experiment:
    """Everything built from a config, before training."""

    config: ExperimentConfig
    source: DatasetSource
    task: Task
    model: nn.Module
    splits: dict[str, list[int]]
    device: torch.device


def build(config: ExperimentConfig, config_dir: Path | None = None) -> Experiment:
    load_all(config.plugins, config_dir)
    repro.seed_everything(config.seed)
    repro.set_deterministic(config.deterministic)

    source = DATASETS.build(config.data.dataset.name, **config.data.dataset.params)
    task = TASKS.build(config.task.name, info=source.info, **config.task.params)
    splits = make_splits(source, config)
    sizing = task.model_kwargs()
    clash = set(sizing) & set(config.model.params)
    if clash:
        raise ValueError(
            f"model.params sets {sorted(clash)}, which the task derives from the dataset"
        )
    model = MODELS.build(config.model.name, **sizing, **config.model.params)
    device = repro.resolve_device(config.train.device)
    return Experiment(config, source, task, model.to(device), splits, device)


def make_splits(source: DatasetSource, config: ExperimentConfig) -> dict[str, list[int]]:
    split = config.data.split
    if split.method == "predefined":
        splits = source.predefined_splits()
        if splits is None:
            raise ValueError(
                f"{config.data.dataset.name} has no predefined split; use split.method: patient"
            )
    else:
        if not source.info.patient_ids_available:
            log.warning("this dataset has no patient IDs, so a patient split is a per-image split")
        splits = patient_split(
            source.patient_ids,
            split.fractions,
            strata=source.strata if split.stratify else None,
            seed=config.seed if split.seed is None else split.seed,
        )
    missing = [name for name in SPLIT_NAMES if name not in splits]
    if missing:
        raise ValueError(f"split lacks {missing}")
    if source.info.patient_ids_available:
        check_disjoint(splits, source.patient_ids)
    else:
        log.warning("patient IDs are unavailable; patient-level leakage cannot be checked")
    return {name: list(splits[name]) for name in SPLIT_NAMES}


def _loader(experiment: Experiment, split: str, train: bool) -> DataLoader:
    config = experiment.config
    transforms = config.data.train_transforms if train else config.data.eval_transforms
    dataset = SplitDataset(
        experiment.source, experiment.splits[split], build_transforms(transforms)
    )
    batch_size = (
        config.data.batch_size if train else (config.data.eval_batch_size or config.data.batch_size)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=config.data.num_workers,
        collate_fn=experiment.task.collate,
        worker_init_fn=repro.worker_init_fn,
        generator=repro.make_generator(config.seed + (0 if train else 1)),
        pin_memory=experiment.device.type == "cuda",
        drop_last=False,
        persistent_workers=config.data.num_workers > 0,
    )


def _optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    o = config.optim
    # No weight decay on biases and normalization parameters.
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    groups = [
        {"params": decay, "weight_decay": o.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if o.name == "adamw":
        return torch.optim.AdamW(groups, lr=o.lr)
    if o.name == "adam":
        return torch.optim.Adam(groups, lr=o.lr)
    return torch.optim.SGD(groups, lr=o.lr, momentum=o.momentum, nesterov=True)


def _scheduler(
    config: ExperimentConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    o, epochs = config.optim, config.train.epochs

    def factor(epoch: int) -> float:
        if epoch < o.warmup_epochs:
            return (epoch + 1) / (o.warmup_epochs + 1)
        if o.scheduler == "cosine":
            span = max(epochs - o.warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * (epoch - o.warmup_epochs) / span))
        if o.scheduler == "step":
            return o.gamma ** ((epoch - o.warmup_epochs) // o.step_size)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def predict(experiment: Experiment, loader: DataLoader) -> list[dict[str, Any]]:
    experiment.model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        predictions.extend(experiment.task.predict(experiment.model, batch, experiment.device))
    return predictions


def _scalar(value: Any) -> float:
    return float(value["value"]) if isinstance(value, dict) else float(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default, allow_nan=True) + "\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_dir(config: ExperimentConfig, git: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = (git.get("commit") or "nogit")[:8]
    base = Path(config.output_dir) / config.name
    run_dir = base / f"{stamp}-{commit}"
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}-{commit}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _attach_log(run_dir: Path) -> logging.Handler:
    handler = logging.FileHandler(run_dir / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("medimage").addHandler(handler)
    return handler


def train(config: ExperimentConfig, config_dir: Path | None = None) -> Path:
    """Train, select the best epoch on validation, evaluate with CIs, and return the run dir."""
    started = repro.utc_now()
    git = repro.git_info(config_dir or Path.cwd())
    experiment = build(config, config_dir)  # fails fast, before a run directory exists
    run_dir = _run_dir(config, git)
    handler = _attach_log(run_dir)
    try:
        return _train(experiment, run_dir, git, started)
    finally:
        logging.getLogger("medimage").removeHandler(handler)
        handler.close()


def _train(experiment: Experiment, run_dir: Path, git: dict[str, Any], started: str) -> Path:
    config = experiment.config
    source, task, model = experiment.source, experiment.task, experiment.model
    dump_config(config, run_dir / "config.yaml")
    run_record = {
        "name": config.name,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "git": git,
        "environment": repro.environment(),
        "device": str(experiment.device),
        "command": sys.argv,
        "dataset": {
            "name": config.data.dataset.name,
            "license": source.info.license,
            "citation": source.info.citation,
            "patient_ids_available": source.info.patient_ids_available,
            "fingerprint": source.fingerprint(),
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "started": started,
    }
    write_json(run_dir / "run.json", run_record)
    ids = source.sample_ids
    write_json(
        run_dir / "splits.json",
        {
            "method": config.data.split.method,
            "summary": summarize(experiment.splits, source.patient_ids, source.strata),
            "ids": {name: [ids[i] for i in idx] for name, idx in experiment.splits.items()},
        },
    )
    log.info("run directory: %s", run_dir)
    log.info("splits: %s", summarize(experiment.splits, source.patient_ids))

    train_loader = _loader(experiment, "train", train=True)
    val_loader = _loader(experiment, "val", train=False)
    optimizer = _optimizer(config, model)
    scheduler = _scheduler(config, optimizer)
    use_amp = config.train.amp and experiment.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    monitor = config.train.monitor or task.default_monitor
    sign = 1.0 if config.train.monitor_mode == "max" else -1.0
    best_score, best_epoch, stale = -math.inf, -1, 0
    checkpoint = run_dir / "checkpoint.pt"

    with (run_dir / "history.jsonl").open("w") as history:
        for epoch in range(config.train.epochs):
            model.train()
            tick, losses = time.perf_counter(), []
            for step, batch in enumerate(train_loader):
                if (
                    config.train.max_steps_per_epoch is not None
                    and step >= config.train.max_steps_per_epoch
                ):
                    break
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(experiment.device.type, enabled=use_amp):
                    loss = task.loss(model, batch, experiment.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"loss became {loss.item()} at epoch {epoch} step {step}"
                    )
                scaler.scale(loss).backward()
                if config.train.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                if config.train.log_every and step % config.train.log_every == 0:
                    log.info("epoch %d step %d loss %.4f", epoch, step, loss.item())
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            val_metrics = task.metrics(predict(experiment, val_loader))
            score = val_metrics.get(monitor)
            if score is None:
                raise KeyError(
                    f"monitor {monitor!r} is not a {config.task.name} metric; choose from {sorted(val_metrics)}"
                )
            entry = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "lr": lr,
                "seconds": round(time.perf_counter() - tick, 2),
                "val": val_metrics,
            }
            history.write(json.dumps(entry, default=_json_default) + "\n")
            history.flush()
            log.info(
                "epoch %d train_loss %.4f val %s %.4f", epoch, entry["train_loss"], monitor, score
            )

            if math.isfinite(score) and sign * score > best_score:
                best_score, best_epoch, stale = sign * score, epoch, 0
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), monitor: score}, checkpoint
                )
            else:
                stale += 1
                if not math.isfinite(score):
                    log.warning("val %s is %s at epoch %d", monitor, score, epoch)
            patience = config.train.early_stopping_patience
            if patience is not None and stale >= patience:
                log.info("early stopping after epoch %d", epoch)
                break

    if best_epoch < 0:
        torch.save({"epoch": epoch, "model": model.state_dict()}, checkpoint)
        log.warning("validation %s was never finite; keeping the last epoch", monitor)
    results = evaluate(experiment, checkpoint, run_dir)
    results["selection"] = {
        "monitor": monitor,
        "mode": config.train.monitor_mode,
        "best_epoch": best_epoch,
    }
    write_json(run_dir / "metrics.json", results)
    run_record["finished"] = repro.utc_now()
    write_json(run_dir / "run.json", run_record)
    log.info("finished: %s", run_dir / "metrics.json")
    return run_dir


def evaluate(experiment: Experiment, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    """Evaluate the checkpoint on the configured splits with bootstrap CIs.

    Calibration (the binary decision threshold) is always fitted on validation predictions,
    then applied unchanged to every split.
    """
    config = experiment.config
    state = torch.load(checkpoint, map_location=experiment.device, weights_only=True)
    experiment.model.load_state_dict(state["model"])
    val_predictions = predict(experiment, _loader(experiment, "val", train=False))
    calibration = experiment.task.calibrate(val_predictions)
    results: dict[str, Any] = {"checkpoint_epoch": state["epoch"], "calibration": calibration}
    settings = {
        "replicates": config.eval.bootstrap_samples,
        "confidence": config.eval.confidence,
        "unit": config.eval.bootstrap_unit,
        "seed": config.seed,
    }
    results["bootstrap"] = settings | {
        "method": "percentile",
        "unit": settings["unit"]
        if experiment.source.info.patient_ids_available
        else "sample (no patient IDs)",
    }
    for split in config.eval.splits:
        predictions = (
            val_predictions
            if split == "val"
            else predict(experiment, _loader(experiment, split, train=False))
        )
        metrics = experiment.task.metrics(
            predictions, calibration, settings if config.eval.bootstrap_samples else None
        )
        results[split] = metrics
        write_rows(
            run_dir / f"predictions_{split}.csv", experiment.task.prediction_rows(predictions)
        )
        summary = {
            k: _scalar(v) for k, v in metrics.items() if isinstance(v, dict) and "/" not in k
        }
        log.info("%s: %s", split, ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    return results


def evaluate_run(run_dir: Path | str, splits: list[str] | None = None) -> dict[str, Any]:
    """Re-evaluate a finished run from its saved config and checkpoint."""
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    if splits:
        config.eval.splits = splits
    experiment = build(config, run_dir)
    return evaluate(experiment, run_dir / "checkpoint.pt", run_dir)
