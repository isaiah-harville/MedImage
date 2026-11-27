"""Papers about published runs, and the table of datasets that have one.

A paper lives in ``papers/<name>/`` with its LaTeX source, the built PDF, and ``paper.yaml``,
which names the published run it reports. The notebook copies each PDF next to the
generated pages and links it from the Papers page and from its run's page.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from opendiagnostics.notebook.model import Project
from opendiagnostics.notebook.pages import GENERATED, dataset_label, fmt, table


class PaperRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str
    description: str
    project: str
    run: str
    pdf: str


@dataclass(frozen=True)
class Paper:
    name: str
    title: str
    description: str
    project: str
    run: str
    source: Path

    @property
    def href(self) -> str:
        """The PDF's path relative to the notebook root."""
        return f"papers/{self.name}.pdf"


def load_papers(papers_dir: Path, projects: Sequence[Project]) -> list[Paper]:
    """Every ``papers/*/paper.yaml``. A paper must name a published run and have its PDF."""
    if not papers_dir.is_dir():
        return []
    runs = {(p.name, r.run_id) for p in projects for r in p.runs}
    papers = []
    for path in sorted(papers_dir.glob("*/paper.yaml")):
        record = PaperRecord.model_validate(yaml.safe_load(path.read_text()))
        source = path.parent / record.pdf
        if (record.project, record.run) not in runs:
            raise ValueError(
                f"{path} names run {record.project}/{record.run}, which is not published"
            )
        if not source.is_file():
            raise ValueError(
                f"{path} names {source}, which does not exist; build it with "
                f"`tectonic -X compile main.tex` in {path.parent}"
            )
        papers.append(
            Paper(
                path.parent.name,
                record.title,
                record.description,
                record.project,
                record.run,
                source,
            )
        )
    return papers


def papers_page(papers: Sequence[Paper], projects: Sequence[Project]) -> str:
    runs = {(p.name, r.run_id): r for p in projects for r in p.runs}
    intro = (
        "Papers on the published runs, as IEEE conference drafts. Each reports one run, and "
        "every number in it comes from that run's committed files.\n\n"
    )
    if not papers:
        return GENERATED + "# Papers\n\n" + intro + "No papers yet.\n"
    sections = [
        f"## [{paper.title}]({paper.href})\n\n{paper.description}\n\n"
        f"Run [{paper.run}]({paper.project}/{paper.run}/index.md) of "
        f"[{paper.project}]({paper.project}/index.md), test "
        f"{fmt(runs[paper.project, paper.run].headline())}. [PDF]({paper.href})\n"
        for paper in papers
    ]
    return GENERATED + "# Papers\n\n" + intro + "\n".join(sections)


def published_table(projects: Sequence[Project], papers: Sequence[Paper]) -> str:
    """Datasets with a published model, for the dataset survey page. Links are relative to
    the docs root, where the survey page is."""
    by_run = {(p.project, p.run): p for p in papers}
    rows = []
    for project in projects:
        if not project.runs:
            continue
        run = project.runs[0]
        paper = by_run.get((project.name, run.run_id))
        rows.append(
            [
                f"[{dataset_label(project.config)}](notebook/{project.name}/index.md)",
                f"{run.monitor} {fmt(run.headline())}",
                f"[results](notebook/{project.name}/{run.run_id}/index.md)",
                f"[PDF](notebook/{paper.href})" if paper else "",
            ]
        )
    if not rows:
        return GENERATED + "No dataset has a published model yet.\n"
    return GENERATED + table(["Dataset", "Test result [95% CI]", "Latest run", "Paper"], rows)
