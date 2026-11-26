import csv

import numpy as np
import pytest
from xpt import expect

from opendiagnostics.cli import main
from opendiagnostics.metrics.compare import compare, read_predictions


def _write(path, ids, patients, target, prob):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "patient_id", "target", "prob/positive"])
        writer.writerows(zip(ids, patients, target, prob, strict=True))
    return read_predictions(path)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, size=400)
    ids = [f"s{i}" for i in range(400)]
    patients = [f"p{i // 2}" for i in range(400)]
    return rng, ids, patients, target


def test_a_model_compared_with_itself_shows_no_difference(tmp_path, labels):
    rng, ids, patients, target = labels
    prob = rng.random(400)
    a = _write(tmp_path / "a.csv", ids, patients, target, prob)
    result = compare(a, a, replicates=200)
    assert result.difference == 0
    assert result.lower == result.upper == 0
    assert result.p_value == 1.0
    assert result.patients == 200


def test_a_better_model_wins_with_an_interval_above_zero(tmp_path, labels):
    rng, ids, patients, target = labels
    weak = _write(tmp_path / "weak.csv", ids, patients, target, rng.random(400) + 0.2 * target)
    strong = _write(tmp_path / "strong.csv", ids, patients, target, rng.random(400) + 1.5 * target)
    result = compare(weak, strong, replicates=500)
    assert result.difference > 0
    assert result.lower > 0
    assert result.p_value < 0.01


def test_rows_are_aligned_by_sample_id_not_file_order(tmp_path, labels):
    rng, ids, patients, target = labels
    prob = rng.random(400) + target
    a = _write(tmp_path / "a.csv", ids, patients, target, prob)
    order = rng.permutation(400)
    b = _write(
        tmp_path / "b.csv",
        [ids[i] for i in order],
        [patients[i] for i in order],
        target[order],
        prob[order],
    )
    assert compare(a, b, replicates=100).difference == 0


def test_runs_on_different_samples_are_refused(tmp_path, labels):
    rng, ids, patients, target = labels
    a = _write(tmp_path / "a.csv", ids, patients, target, rng.random(400))
    b = _write(tmp_path / "b.csv", ids[:300], patients[:300], target[:300], rng.random(300))
    with pytest.raises(ValueError, match="same split"):
        compare(a, b)


def test_cli_reports_the_difference_interval_and_p_value(tmp_path, labels, capsys):
    rng, ids, patients, target = labels
    _write(tmp_path / "a.csv", ids, patients, target, np.round(rng.random(400) + 0.3 * target, 6))
    _write(tmp_path / "b.csv", ids, patients, target, np.round(rng.random(400) + 1.0 * target, 6))
    assert (
        main(["compare", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--replicates", "200"])
        == 0
    )
    expect(
        capsys.readouterr().out,
        """auroc: A 0.7800, B 1.0000
B - A = +0.2200 [95% CI +0.1847, +0.2596], p < 0.0050
400 samples from 200 patients, 200 of 200 replicates valid
""",
    )
