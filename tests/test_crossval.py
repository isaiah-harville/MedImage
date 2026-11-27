import json

import numpy as np
import pytest

from opendiagnostics.crossval import CrossValidationError, out_of_fold_threshold
from opendiagnostics.data.base import kfold_split
from opendiagnostics.metrics import classification as cm


def test_kfold_keeps_patients_together_covers_the_pool_and_leaves_test_alone():
    patients = [f"p{i // 3}" for i in range(60)]
    strata = [i // 3 % 2 for i in range(60)]
    pool, test = list(range(48)), list(range(48, 60))
    held = []
    for fold in range(4):
        split = kfold_split(pool, test, patients, strata, 4, fold, seed=0)
        assert split["test"] == test
        assert set(split["train"]) | set(split["val"]) == set(pool)
        assert not {patients[i] for i in split["train"]} & {patients[i] for i in split["val"]}
        held += split["val"]
    assert sorted(held) == pool


def test_kfold_refuses_a_fold_out_of_range():
    with pytest.raises(ValueError, match=r"fold must be in 0\.\.2"):
        kfold_split([0, 1, 2], [], ["a", "b", "c"], None, 3, 3)


def _run(path, ids, val_rows=None):
    path.mkdir()
    (path / "splits.json").write_text(json.dumps({"ids": ids}))
    if val_rows is not None:
        lines = ["id,patient_id,target,prob/positive"]
        lines += [f"{i},{i},{t},{p}" for i, t, p in val_rows]
        (path / "predictions_val.csv").write_text("\n".join(lines) + "\n")
    return path


def _folds(tmp_path, rng):
    ids = [f"s{i}" for i in range(40)]
    target = rng.integers(0, 2, 40)
    score = np.clip(target * 0.4 + rng.random(40) * 0.6, 0, 1)
    run = _run(tmp_path / "run", {"train": ids[:30], "val": ids[30:], "test": ["t0", "t1"]})
    folds = [
        _run(
            tmp_path / f"fold{k}",
            {"train": [], "val": ids[k::4], "test": ["t0", "t1"]},
            [(ids[i], int(target[i]), float(score[i])) for i in range(k, 40, 4)],
        )
        for k in range(4)
    ]
    return run, folds, target, score


def test_the_threshold_is_fitted_on_all_out_of_fold_predictions(tmp_path):
    run, folds, target, score = _folds(tmp_path, np.random.default_rng(0))
    calibration = out_of_fold_threshold(run, folds)
    order = [i for k in range(4) for i in range(k, 40, 4)]
    assert calibration["threshold"] == pytest.approx(
        cm.youden_threshold(target[order], score[order])
    )
    assert calibration["samples"] == 40
    assert calibration["folds"] == ["fold0", "fold1", "fold2", "fold3"]


def test_folds_that_miss_samples_or_touch_test_are_refused(tmp_path):
    run, folds, _, _ = _folds(tmp_path, np.random.default_rng(1))
    with pytest.raises(CrossValidationError, match="needs exactly one out-of-fold prediction"):
        out_of_fold_threshold(run, folds[:3])
    leaky = _run(tmp_path / "leaky", {"train": [], "val": ["t0"], "test": []})
    with pytest.raises(CrossValidationError, match="test split"):
        out_of_fold_threshold(run, [*folds, leaky])
