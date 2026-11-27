"""Run a queue of training jobs on one GPU while their datasets download in parallel.

    uv run python scripts/run_queue.py queue.yaml [--downloads 3] [--logs queue-logs]

``queue.yaml`` lists jobs in the order the GPU should run them::

    jobs:
      - name: organmnist3d                     # a unique name, used for logs and status
        config: models/organmnist3d-densenet/config.yaml
        args: ["--trainer.max_epochs=60"]      # optional overrides for `opendiagnostics fit`
        prepare: [uv, run, python, scripts/datasets/nih_chestxray14.py, data/nih]
      - name: breastmnist-threshold
        run: [uv, run, opendiagnostics, evaluate, runs/breastmnist-convnext/<run>]

A job with ``config`` trains it with ``opendiagnostics fit``. Its data is prepared first by
``prepare`` if given, otherwise by ``opendiagnostics prepare -c <config>``; every job's
prepare step starts at once, up to ``--downloads`` at a time, so downloads overlap training.
A job with ``run`` runs that command in the GPU lane, in order. A job waits only for its own
prepare step. A failed job is recorded and the queue moves on.

Status goes to ``<logs>/queue-status.json`` after every job, and each step logs to
``<logs>/<name>-<step>.log``. Rerunning the same queue skips jobs that already succeeded.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

FINISHED = re.compile(r"finished: (\S+)metrics\.json")


def load_jobs(path: Path) -> list[dict[str, Any]]:
    jobs = (yaml.safe_load(path.read_text()) or {}).get("jobs") or []
    names = set()
    for job in jobs:
        name = job.get("name")
        if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise SystemExit(f"{path}: every job needs a name of letters, digits, . _ -: {job}")
        if name in names:
            raise SystemExit(f"{path}: job name {name!r} appears twice")
        if ("config" in job) == ("run" in job):
            raise SystemExit(f"{path}: job {name!r} needs exactly one of config or run")
        names.add(name)
    return jobs


def run_step(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode


def run_dir_from(log: Path) -> str | None:
    matches = FINISHED.findall(log.read_text(errors="replace")) if log.exists() else []
    return matches[-1].rstrip("/\\") if matches else None


def prepare_command(job: dict[str, Any]) -> list[str] | None:
    if "run" in job:
        return None
    return job.get("prepare") or ["uv", "run", "opendiagnostics", "prepare", "-c", job["config"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("queue", type=Path)
    parser.add_argument("--downloads", type=int, default=3, help="prepare steps at once")
    parser.add_argument("--logs", type=Path, default=Path("queue-logs"))
    args = parser.parse_args(argv)
    jobs = load_jobs(args.queue)
    status_path = args.logs / "queue-status.json"
    status: dict[str, Any] = json.loads(status_path.read_text()) if status_path.exists() else {}

    def save() -> None:
        args.logs.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    pending = [job for job in jobs if status.get(job["name"], {}).get("state") != "done"]
    with ThreadPoolExecutor(max_workers=max(1, args.downloads)) as pool:
        prepared: dict[str, Future[int] | None] = {}
        for job in pending:
            command = prepare_command(job)
            log = args.logs / f"{job['name']}-prepare.log"
            prepared[job["name"]] = pool.submit(run_step, command, log) if command else None
        for job in pending:
            name = job["name"]
            future = prepared[name]
            if future is not None:
                status[name] = {"state": "preparing"}
                save()
                code = future.result()
                if code != 0:
                    status[name] = {"state": "failed", "step": "prepare", "exit": code}
                    save()
                    print(f"{name}: prepare failed (exit {code})", file=sys.stderr)
                    continue
            started = time.time()
            status[name] = {"state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save()
            log = args.logs / f"{name}-{'fit' if 'config' in job else 'run'}.log"
            command = (
                ["uv", "run", "opendiagnostics", "fit", "-c", job["config"], *job.get("args", [])]
                if "config" in job
                else job["run"]
            )
            code = run_step(command, log)
            status[name] = {
                "state": "done" if code == 0 else "failed",
                "exit": code,
                "minutes": round((time.time() - started) / 60, 1),
                "run_dir": run_dir_from(log),
            }
            save()
            print(f"{name}: {status[name]['state']} (exit {code})")
    failed = [n for n, s in status.items() if s.get("state") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
