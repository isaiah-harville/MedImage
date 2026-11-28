"""Write the SynapseMNIST3D paper's tables from generated/seeds.json and the model projects.

    uv run python -I papers/synapsemnist3d/analysis/tables.py

Run analysis/paper_numbers.py first. Writes generated/tables/seeds.tex, published.tex and
subsets.tex, and generated/subsets.tex (macros). The subsets table sets the first run on each
MedMNIST 3D subset against the best published AUROC in that project's baselines.yaml, both
as the list stood when SynapseMNIST3D was chosen (commit CHOSEN) and as it stands now.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
TABLES = PAPER / "generated/tables"
CHOSEN = "39e65a0"
SUBSETS = ("adrenal", "fracture", "nodule", "organ", "synapse", "vessel")
SEED_NAMES = ("Zero", "One", "Two", "Three", "Four")
PUBLISHED = (
    (
        "R3D-18, Kinetics-400 pretrained, fine-tuned (64³)",
        "R3D-18, Kinetics-400~\\cite{nuwagira2026topocam}",
        "64",
        "n.s.",
    ),
    (
        "DINOv3 ViT-L/16, fully fine-tuned, 3D baseline (mean of 5 runs)",
        "ViT-L/16, DINOv3, 3D~\\cite{yu2026planecycle}",
        "n.s.",
        "mean of 5",
    ),
    (
        "Axial-centric cross-plane attention (64³)",
        "MedDINOv3 slices, frozen~\\cite{park2026axial}",
        "64",
        "n.s.",
    ),
    (
        "MTDL, Hodge-decomposition channels (64³), mean of 3 seeds",
        "MTDL~\\cite{liu2026manifold}",
        "64",
        "mean of 3",
    ),
    (
        "CNMTDL, combinatorial-complex MTDL (64³), mean of 5 seeds",
        "CNMTDL~\\cite{wachira2026combinatorial}",
        "64",
        "mean of 5",
    ),
    (
        "ResNet-50 + 3D (28³), mean of at least 3 trials",
        "ResNet-50 + 3D~\\cite{yang2023medmnist}",
        "28",
        "mean of $\\geq$3",
    ),
)


def ci(entry: dict[str, float], digits: int = 4) -> str:
    return f"{entry['value']:.{digits}f} [{entry['lower']:.{digits}f}, {entry['upper']:.{digits}f}]"


def table(
    label: str, caption: str, spec: str, header: str, body: list[str], wide: bool = False
) -> str:
    env = "table*" if wide else "table"
    return (
        f"\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n"
        "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        + " \\\\\n".join(body)
        + f" \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n"
    )


def seeds_table(data: dict) -> str:
    body = []
    for index, seed in enumerate(SEED_NAMES):
        scratch, kinetics = data["runs"][f"Scratch{seed}"], data["runs"][f"Kinetics{seed}"]
        pair = data["paired"][seed]
        body.append(
            f"{index} & {ci(scratch['test']['auroc'])} & {scratch['test']['accuracy_at_0.5']['value']:.3f} & "
            f"{ci(kinetics['test']['auroc'])} & {kinetics['test']['accuracy_at_0.5']['value']:.3f} & "
            f"{pair['difference']:+.4f} [{pair['lower']:+.4f}, {pair['upper']:+.4f}] & "
            + " & ".join(
                f"{v:.3f}" if v >= 0.001 else "$<$0.001" for v in (pair["p_value"], pair["holm"])
            )
        )
    return table(
        "tab:seeds",
        "Do the two networks differ on each seed? Test AUROC with its 95\\% bootstrap interval over "
        "the 352 test volumes, accuracy at the 0.5 threshold (ACC), and the paired difference "
        "R3D-18 minus ResNet-18 from \\texttt{opendiagnostics compare} (2000 paired replicates) "
        "with its two-sided $p$ and the Holm-adjusted $p$ over the five seeds. Runs are paired "
        "by seed number only; each test is paired over the test volumes.",
        "@{}clllllll@{}",
        "Seed & ResNet-18, scratch & ACC & R3D-18, Kinetics & ACC & Difference [95\\% CI] & $p$ & Holm",
        body,
        wide=True,
    )


def published_table(data: dict) -> str:
    body = []
    for method, label, size, runs in PUBLISHED:
        metrics = data["baselines"][method]
        acc = f"{metrics['accuracy']:.3f}" if "accuracy" in metrics else "--"
        body.append(f"{label} & {size} & {runs} & {metrics['auroc']:.3f} & {acc}")
    return table(
        "tab:published",
        "Which published SynapseMNIST3D results do we compare with? Test AUROC and accuracy "
        "(ACC) on the official split as each paper reports them; none states its ACC threshold. Size: input edge in voxels "
        "(n.s.: not stated). Runs: how many training runs the value summarizes. None reports an "
        "SD in its main text, an interval, or per-volume predictions.",
        "@{}lccll@{}",
        "Method & Size & Runs & AUROC & ACC",
        body,
    )


def best(baselines: list[dict]) -> float:
    return max(b["metrics"]["auroc"] for b in baselines if "auroc" in b["metrics"])


def verdict(entry: dict[str, float], value: float) -> str:
    if entry["lower"] > value:
        return "above"
    return "below" if entry["upper"] < value else "inside"


def subsets_table() -> tuple[str, dict[str, str]]:
    body, above_then, commits = [], 0, set()
    for subset in SUBSETS:
        project = f"models/{subset}mnist3d-resnet18"
        first = sorted((ROOT / project / "runs").iterdir())[0]
        commits.add(first.name[-8:])
        entry = json.loads((first / "metrics.json").read_text())["test"]["auroc"]
        then = best(
            yaml.safe_load(
                subprocess.run(
                    ["git", "show", f"{CHOSEN}:{project}/baselines.yaml"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )["baselines"]
        )
        now = best(yaml.safe_load((ROOT / project / "baselines.yaml").read_text())["baselines"])
        above_then += verdict(entry, then) == "above"
        name = "OrganMNIST3D" if subset == "organ" else f"{subset.capitalize()}MNIST3D"
        body.append(
            f"{name} & {ci(entry, 3)} & {then:.3f} & {verdict(entry, then)} & {now:.3f} & {verdict(entry, now)}"
        )
    text = table(
        "tab:subsets",
        "How did the first SynapseMNIST3D run compare with the other MedMNIST 3D subsets? Test "
        "AUROC [95\\% CI] of the first run of the same from-scratch ResNet-18 recipe on each "
        "subset at 64 voxels, against the best published value in that project's list when "
        f"SynapseMNIST3D was chosen (commit {CHOSEN}) and now. Only the SynapseMNIST3D list has "
        "had a literature review since. Our CI: whether our interval lies above or below that "
        "value, or contains it.",
        "@{}llllll@{}",
        "Subset & Ours [95\\% CI] & Then & Our CI & Now & Our CI",
        body,
        wide=True,
    )
    macros = {"subsetsTotal": str(len(SUBSETS)), "subsetsAboveThen": str(above_then)}
    return text, macros


def main() -> None:
    data = json.loads((PAPER / "generated/seeds.json").read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "seeds.tex").write_text(seeds_table(data))
    (TABLES / "published.tex").write_text(published_table(data))
    text, macros = subsets_table()
    (TABLES / "subsets.tex").write_text(text)
    (PAPER / "generated/subsets.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )
    print(text, macros)


if __name__ == "__main__":
    main()
