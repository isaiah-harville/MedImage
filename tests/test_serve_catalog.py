from opendiagnostics.serve.catalog import DATASETS, ORDER, describe, rank


def test_every_described_dataset_has_a_category():
    assert all(scan.category != "Other" for scan in DATASETS.values())


def test_a_project_is_described_by_its_dataset():
    assert describe("kermany-oct-convnext").category == "Eye"
    assert describe("pneumoniamnist-cnn") == describe("pneumoniamnist-convnext")
    assert describe("new-dataset-convnext").category == "Other"


def test_models_sort_by_category_then_dataset():
    ids = ["new-dataset-convnext", "dermamnist-convnext", "cmmd-convnext", "chestmnist-convnext"]
    assert sorted(ids, key=rank) == [
        "chestmnist-convnext",
        "cmmd-convnext",
        "dermamnist-convnext",
        "new-dataset-convnext",
    ]
    assert ORDER[0] == "CT and MRI"
    assert ORDER[-1] == "Other"


def test_every_published_classifier_is_described(root):
    projects = [
        p.name
        for p in (root / "models").iterdir()
        if (p / "config.yaml").is_file()
        and "ImageClassifier" in (p / "config.yaml").read_text()
        and any((p / "runs").glob("*/metrics.json"))
    ]
    assert projects
    assert [p for p in projects if describe(p).category == "Other"] == []
