import math

import numpy as np
import pytest
from xpt import expect

from opendiagnostics.cli import main
from opendiagnostics.evaluation import write_rows
from opendiagnostics.metrics import froc


def nodule(center, diameter):
    return [*(c - diameter / 2 for c in center), *(c + diameter / 2 for c in center)]


def detection(center, size=4.0):
    return nodule(center, size)


@pytest.fixture
def scans():
    """Five scans worked by hand below."""
    return [
        froc.Scan(
            id="a",
            patient_id="p1",
            boxes=np.array(
                [
                    detection((1, 1, 1)),
                    detection((2, 0, 0)),
                    detection((50, 1, 0)),
                    detection((5, 0, 0)),
                ]
            ),
            scores=np.array([0.9, 0.6, 0.8, 0.3]),
            truth=np.array([nodule((0, 0, 0), 10)]),
            ignore=np.array([nodule((50, 0, 0), 10)]),
        ),
        froc.Scan(
            id="b",
            patient_id="p2",
            boxes=np.array([detection((0, 0, 0))]),
            scores=np.array([0.5]),
            truth=np.array([nodule((100, 100, 100), 6)]),
            ignore=np.zeros((0, 6)),
        ),
        froc.Scan(
            id="c",
            patient_id="p3",
            boxes=np.array([detection((30, 30, 30)), detection((-40, 0, 0))]),
            scores=np.array([0.95, 0.7]),
            truth=np.zeros((0, 6)),
            ignore=np.zeros((0, 6)),
        ),
        froc.Scan(
            id="d",
            patient_id="p3",
            boxes=np.array([detection((0, 0, 1))]),
            scores=np.array([0.4]),
            truth=np.array([nodule((0, 0, 0), 10)]),
            ignore=np.zeros((0, 6)),
        ),
        froc.Scan(
            id="e",
            patient_id="p4",
            boxes=np.zeros((0, 6)),
            scores=np.zeros(0),
            truth=np.zeros((0, 6)),
            ignore=np.zeros((0, 6)),
        ),
    ]


def _hits(scans):
    return [froc.match_scan(s.boxes, s.scores, s.truth, s.ignore) for s in scans]


def test_matching_follows_the_luna16_rules(scans):
    first = froc.match_scan(scans[0].boxes, scans[0].scores, scans[0].truth, scans[0].ignore)
    # 0.9 and 0.6 hit the nodule: it counts once, at 0.9, and 0.6 is dropped.
    assert first.lesion_scores.tolist() == [0.9]
    # 0.8 hits the ignored finding; 0.3 sits exactly on the sphere's edge, which is a miss.
    assert first.false_positive_scores.tolist() == [0.3]
    assert first.lesions == 1


def test_cpm_matches_a_hand_worked_froc_curve(scans):
    # Detected lesions score 0.9 and 0.4 out of 3 lesions; false positives score 0.95, 0.7,
    # 0.5 and 0.3 over 5 scans. Descending thresholds give (FPs per scan, sensitivity):
    # (0.2, 0), (0.2, 1/3), (0.4, 1/3), (0.6, 1/3), (0.6, 2/3), (0.8, 2/3).
    metrics = froc.froc_metrics(_hits(scans))
    third = 1 / 3
    expected = [0.0, third, third, 2 * third, 2 * third, 2 * third, 2 * third]
    got = [metrics[f"sensitivity_at_{r:g}_fp"] for r in froc.FP_RATES]
    assert got == pytest.approx(expected)
    assert metrics["cpm"] == pytest.approx(10 / 21)


def test_a_detection_inside_the_box_corner_but_outside_the_sphere_misses():
    box = nodule((0, 0, 0), 10)
    assert froc.inside([[4.9, 0, 0], [2.8, 2.8, 2.8], [3, 3, 3]], [box]).ravel().tolist() == [
        True,
        True,
        False,
    ]


def test_cpm_is_undefined_without_lesions_and_zero_without_hits(scans):
    assert math.isnan(froc.froc_metrics(_hits(scans[2:3]))["cpm"])
    assert froc.froc_metrics(_hits(scans[1:2]))["cpm"] == 0.0


def test_a_system_without_false_positives_reaches_its_sensitivity_at_every_rate(scans):
    metrics = froc.froc_metrics(_hits(scans[3:4]))
    assert metrics["cpm"] == 1.0


def _write(path, scans):
    write_rows(path, [row for s in scans for row in froc.scan_rows(s, "nodule")])


def test_prediction_rows_round_trip_through_csv(scans, tmp_path):
    _write(tmp_path / "p.csv", scans)
    read = froc.read_scans(tmp_path / "p.csv")
    assert [s.id for s in read] == [s.id for s in scans]
    for a, b in zip(read, scans, strict=True):
        assert a.patient_id == b.patient_id
        np.testing.assert_allclose(a.boxes, b.boxes)
        np.testing.assert_allclose(a.scores, b.scores)
        np.testing.assert_allclose(a.truth, b.truth)
        np.testing.assert_allclose(a.ignore, b.ignore)


def test_pooled_folds_score_like_one_file_and_refuse_a_scan_twice(scans, tmp_path):
    _write(tmp_path / "fold0.csv", scans[:2])
    _write(tmp_path / "fold1.csv", scans[2:])
    pooled = froc.pool([tmp_path / "fold0.csv", tmp_path / "fold1.csv"])
    assert froc.froc_metrics(_hits(pooled))["cpm"] == pytest.approx(10 / 21)
    with pytest.raises(ValueError, match="scan a is in both"):
        froc.pool([tmp_path / "fold0.csv", tmp_path / "fold0.csv"])


def test_cli_reports_pooled_cpm_with_a_patient_interval(scans, tmp_path, capsys):
    _write(tmp_path / "fold0.csv", scans[:2])
    _write(tmp_path / "fold1.csv", scans[2:])
    args = ["froc", str(tmp_path / "fold0.csv"), str(tmp_path / "fold1.csv")]
    assert main([*args, "--replicates", "200"]) == 0
    expect(
        capsys.readouterr().out,
        """cpm: 0.4762 [95% CI 0.0000, 1.0000]
sensitivity_at_0.125_fp: 0.0000 [95% CI 0.0000, 1.0000]
sensitivity_at_0.25_fp: 0.3333 [95% CI 0.0000, 1.0000]
sensitivity_at_0.5_fp: 0.3333 [95% CI 0.0000, 1.0000]
sensitivity_at_1_fp: 0.6667 [95% CI 0.0000, 1.0000]
sensitivity_at_2_fp: 0.6667 [95% CI 0.0000, 1.0000]
sensitivity_at_4_fp: 0.6667 [95% CI 0.0000, 1.0000]
sensitivity_at_8_fp: 0.6667 [95% CI 0.0000, 1.0000]
5 scans from 4 patients, 3 lesions
""",
    )
