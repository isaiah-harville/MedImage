import pytest
from xpt import expect

from opendiagnostics.data import LeakageError, check_disjoint, patient_split, summarize


def test_patients_never_span_splits():
    patients = [f"p{i // 3}" for i in range(300)]
    strata = [(i // 3) % 2 for i in range(300)]
    splits = patient_split(patients, (0.6, 0.2, 0.2), strata=strata, seed=1)
    check_disjoint(splits, patients)
    assert sorted(i for s in splits.values() for i in s) == list(range(300))
    by_split = {name: {patients[i] for i in idx} for name, idx in splits.items()}
    assert [len(by_split[n]) for n in ("train", "val", "test")] == [60, 20, 20]


def test_stratification_balances_prevalence():
    patients = [f"p{i}" for i in range(1000)]
    strata = [int(i % 10 == 0) for i in range(1000)]  # 10% prevalence
    splits = patient_split(patients, strata=strata, seed=0)
    for idx in splits.values():
        prevalence = sum(strata[i] for i in idx) / len(idx)
        assert prevalence == pytest.approx(0.1, abs=0.01)


def test_split_is_deterministic_for_a_seed():
    patients = [f"p{i % 50}" for i in range(200)]
    assert patient_split(patients, seed=3) == patient_split(patients, seed=3)
    assert patient_split(patients, seed=3) != patient_split(patients, seed=4)


def test_leakage_is_detected():
    with pytest.raises(LeakageError, match="patient 'a'"):
        check_disjoint({"train": [0], "test": [1]}, ["a", "a"])


def test_bad_fractions():
    with pytest.raises(ValueError, match="sum to 1"):
        patient_split(["a", "b"], (0.5, 0.5, 0.5))


def test_summary_counts_samples_patients_and_labels():
    patients = ["a", "a", "b", "c", "c", "d"]
    strata = [1, 1, 0, 0, 0, 1]
    expect(
        summarize({"train": [0, 1, 2], "test": [3, 4, 5]}, patients, strata),
        {
            "train": {"samples": 3, "patients": 2, "strata": {"0": 1, "1": 2}},
            "test": {"samples": 3, "patients": 2, "strata": {"0": 2, "1": 1}},
        },
    )
