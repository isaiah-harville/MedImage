import csv
import json

from conftest import fit_smoke

from opendiagnostics.cli import main
from opendiagnostics.metrics import froc


def test_3d_detection_smoke_config_trains_and_scores_cpm(tmp_path, capsys):
    run_dir = fit_smoke("detection_3d", tmp_path)
    names = {p.name for p in run_dir.iterdir()}
    assert names >= {
        "metrics.json",
        "predictions_val.csv",
        "predictions_test.csv",
        "checkpoint.ckpt",
    }
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cpm = metrics["test"]["cpm"]
    assert cpm["lower"] <= cpm["value"] <= cpm["upper"]
    assert cpm["unit"] == "patient"
    assert metrics["selection"]["monitor"] == "val/cpm"
    scans = froc.read_scans(run_dir / "predictions_test.csv")
    assert len(scans) == metrics["test"]["n_scans"]
    assert sum(len(s.truth) for s in scans) == metrics["test"]["n_lesions"]
    with (run_dir / "metrics.csv").open() as handle:
        epochs = {row["epoch"] for row in csv.DictReader(handle) if row["epoch"]}
    assert len(epochs) == 2
    capsys.readouterr()
    assert main(["froc", str(run_dir / "predictions_test.csv"), "--json"]) == 0
    pooled = json.loads(capsys.readouterr().out)
    assert pooled["cpm"]["value"] == cpm["value"]
