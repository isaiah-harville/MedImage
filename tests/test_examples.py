from xpt import expect

from opendiagnostics.examples import choose_classification


def _binary_rows():
    return [
        {"id": "a", "patient_id": "a", "target": "1", "prob/pneumonia": "0.95"},
        {"id": "b", "patient_id": "b", "target": "1", "prob/pneumonia": "0.20"},
        {"id": "c", "patient_id": "c", "target": "0", "prob/pneumonia": "0.05"},
        {"id": "d", "patient_id": "d", "target": "0", "prob/pneumonia": "0.90"},
        {"id": "e", "patient_id": "e", "target": "1", "prob/pneumonia": "0.70"},
    ]


def test_binary_examples_take_confident_correct_then_confident_wrong_per_class():
    picks = choose_classification(_binary_rows(), ["normal", "pneumonia"], 0.5, limit=8)
    expect(
        [(p["id"], p["truth"], p["prediction"], p["correct"]) for p in picks],
        [
            ("c", "normal", "normal", True),
            ("a", "pneumonia", "pneumonia", True),
            ("d", "normal", "pneumonia", False),
            ("b", "pneumonia", "normal", False),
            ("e", "pneumonia", "pneumonia", True),
        ],
    )


def test_examples_stop_at_the_limit():
    picks = choose_classification(_binary_rows(), ["normal", "pneumonia"], 0.5, limit=2)
    assert [p["id"] for p in picks] == ["c", "a"]


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
    assert [(p["id"], p["label"], p["correct"]) for p in picks] == [("x", 2, True), ("y", 1, False)]


def test_multilabel_examples_show_the_most_confident_positive_per_label():
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
    assert [(p["id"], p["label"], p["truth"], p["prediction"]) for p in picks] == [
        ("x", 0, "mass", "mass"),
        ("y", 1, "mass, effusion", "effusion"),
    ]
