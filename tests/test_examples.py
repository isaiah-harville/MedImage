from opendiagnostics.examples import choose_classification


def _binary_rows():
    return [
        {"id": "a", "patient_id": "a", "target": "1", "prob/pneumonia": "0.95"},
        {"id": "b", "patient_id": "b", "target": "1", "prob/pneumonia": "0.20"},
        {"id": "c", "patient_id": "c", "target": "0", "prob/pneumonia": "0.05"},
        {"id": "d", "patient_id": "d", "target": "0", "prob/pneumonia": "0.90"},
        {"id": "e", "patient_id": "e", "target": "1", "prob/pneumonia": "0.70"},
    ]


def _many_rows(n=40):
    return [
        {"id": f"s{i:02d}", "patient_id": f"s{i:02d}", "target": str(i % 2), "prob/x": str(i / n)}
        for i in range(n)
    ]


def test_examples_alternate_classes_until_the_limit():
    picks = choose_classification(_many_rows(), ["no", "yes"], 0.5, limit=8)
    assert [p["truth"] for p in picks] == ["no", "yes"] * 4
    assert len({p["id"] for p in picks}) == 8


def test_examples_are_a_seeded_sample():
    rows = _many_rows()
    first = choose_classification(rows, ["no", "yes"], 0.5, limit=8)
    assert choose_classification(list(reversed(rows)), ["no", "yes"], 0.5, limit=8) == first
    assert choose_classification(rows, ["no", "yes"], 0.5, limit=8, seed=1) != first


def test_examples_do_not_favour_confident_cases():
    rows = _many_rows(400)
    picks = choose_classification(rows, ["no", "yes"], 0.5, limit=200)
    population = [max(float(r["prob/x"]), 1 - float(r["prob/x"])) for r in rows]
    sample = [p["confidence"] for p in picks]
    assert abs(sum(sample) / len(sample) - sum(population) / len(population)) < 0.05


def test_binary_examples_record_the_thresholded_prediction():
    picks = choose_classification(_binary_rows(), ["normal", "pneumonia"], 0.5, limit=8)
    assert len(picks) == 5
    by_id = {p["id"]: (p["truth"], p["prediction"], p["correct"]) for p in picks}
    assert by_id == {
        "a": ("pneumonia", "pneumonia", True),
        "b": ("pneumonia", "normal", False),
        "c": ("normal", "normal", True),
        "d": ("normal", "pneumonia", False),
        "e": ("pneumonia", "pneumonia", True),
    }


def test_multiclass_examples_explain_the_predicted_class():
    rows = [
        {
            "id": "x",
            "patient_id": "x",
            "target": "2",
            "prob/a": "0.1",
            "prob/b": "0.2",
            "prob/c": "0.7",
        },
        {
            "id": "y",
            "patient_id": "y",
            "target": "0",
            "prob/a": "0.2",
            "prob/b": "0.6",
            "prob/c": "0.2",
        },
    ]
    picks = choose_classification(rows, ["a", "b", "c"], 0.5, limit=8)
    assert sorted((p["id"], p["label"], p["correct"]) for p in picks) == [
        ("x", 2, True),
        ("y", 1, False),
    ]


def test_multilabel_examples_take_a_sample_with_each_label():
    rows = [
        {
            "id": "x",
            "patient_id": "x",
            "target/mass": "1",
            "target/effusion": "0",
            "prob/mass": "0.9",
            "prob/effusion": "0.1",
        },
        {
            "id": "y",
            "patient_id": "y",
            "target/mass": "1",
            "target/effusion": "1",
            "prob/mass": "0.4",
            "prob/effusion": "0.8",
        },
    ]
    picks = choose_classification(rows, ["mass", "effusion"], 0.5, limit=8)
    assert sorted((p["id"], p["truth"], p["prediction"]) for p in picks) == [
        ("x", "mass", "mass"),
        ("y", "mass, effusion", "effusion"),
    ]
