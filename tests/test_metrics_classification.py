import numpy as np
import pytest
from sklearn import metrics as sk

from opendiagnostics.metrics import auprc, auroc, bootstrap, threshold_metrics, youden_threshold
from opendiagnostics.metrics.bootstrap import cluster_indices


@pytest.mark.parametrize("seed", range(5))
def test_ranking_metrics_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=300)
    # Rounding creates many ties. torchmetrics computes in float32, hence 1e-6.
    score = np.round(rng.normal(size=300) + y * 0.8, 1)
    assert auroc(y, score) == pytest.approx(sk.roc_auc_score(y, score), abs=1e-6)
    assert auprc(y, score) == pytest.approx(sk.average_precision_score(y, score), abs=1e-6)


def test_undefined_metrics_are_nan():
    assert np.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(auprc([0, 0], [0.1, 0.2]))


def test_youden_threshold_separates_perfectly():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = youden_threshold(y, score)
    result = threshold_metrics(y, score, threshold)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0


def test_bootstrap_interval_covers_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    score = rng.normal(size=200) + y

    def statistic(index):
        return {"auroc": auroc(y[index], score[index])}

    first = bootstrap(statistic, 200, replicates=300, seed=7)["auroc"]
    second = bootstrap(statistic, 200, replicates=300, seed=7)["auroc"]
    assert first == second
    assert first.lower < first.value < first.upper
    assert first.valid_replicates == 300


def test_cluster_bootstrap_is_wider_for_correlated_samples():
    # Each patient contributes 10 identical images, so there are really only 30 observations.
    rng = np.random.default_rng(1)
    y_patient = rng.integers(0, 2, size=30)
    s_patient = rng.normal(size=30) + y_patient
    y, score = np.repeat(y_patient, 10), np.repeat(s_patient, 10)
    groups = [f"p{i}" for i in np.repeat(np.arange(30), 10)]

    def statistic(index):
        return {"auroc": auroc(y[index], score[index])}

    naive = bootstrap(statistic, 300, replicates=500, seed=0)["auroc"]
    cluster = bootstrap(statistic, 300, groups=groups, replicates=500, seed=0)["auroc"]
    assert cluster.unit == "patient"
    assert (cluster.upper - cluster.lower) > 1.5 * (naive.upper - naive.lower)


def test_cluster_indices_keep_groups_together():
    clusters = cluster_indices(["a", "b", "a", "c"])
    assert [c.tolist() for c in clusters] == [[0, 2], [1], [3]]
