"""CAMELYON16 and CAMELYON17 sentinel lymph node slides, as cached feature bags.

Both are CC0 and open. With ``download: true`` (the default), ``prepare_data()`` fetches the
label and checksum files, then streams each slide from the AWS open data bucket into
``root/images/``, encodes it and deletes it (see :class:`SlideFeatureDataModule`). Slides
already under ``root`` with the bucket's layout are used as they are.

Citations: Ehteshami Bejnordi B. et al. JAMA 318(22):2199-2210 (2017); Bandi P. et al. IEEE
TMI 38(2):550-560 (2019); Litjens G. et al. GigaScience 7(6):giy065 (2018).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar

from opendiagnostics.data import camelyon_download as bucket
from opendiagnostics.data.base import DatasetInfo
from opendiagnostics.data.slides import Slide, SlideFeatureDataModule

DATASET_CITATION = (
    "Litjens G. et al. 1399 H&E-stained sentinel lymph node sections of breast cancer "
    "patients: the CAMELYON dataset. GigaScience 7(6):giy065 (2018). "
    "https://doi.org/10.1093/gigascience/giy065"
)
CAMELYON17_CLASSES = ("negative", "itc", "micro", "macro")


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; download the dataset with scripts/datasets/")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def camelyon16_slides(root: Path) -> list[Slide]:
    """Every slide in ``evaluation/reference.csv``: ``normal_*`` and ``tumor_*`` slides are
    the official training set and ``test_*`` the official test set (129 slides)."""
    slides = []
    for row in _rows(root / "evaluation" / "reference.csv"):
        name = Path(row["image"]).stem
        slides.append(
            Slide(
                slide_id=name,
                path=root / "images" / row["image"],
                patient_id=name,
                label=int(row["type"] == "tumor"),
                split="test" if name.startswith("test_") else "train",
                stratum=f"{row['class']}/{row['center']}",
            )
        )
    return slides


def camelyon17_slides(root: Path) -> list[Slide]:
    """The 500 labelled training slides in ``stages.csv``, five per patient.

    Each slide's stratum is its patient's pN stage, so a patient split keeps the stage mix.
    The 500 test slides have no public labels and are not used.
    """
    rows = _rows(root / "stages.csv")
    stages = {Path(r["patient"]).stem: r["stage"] for r in rows if r["patient"].endswith(".zip")}
    slides = []
    for row in rows:
        if not row["patient"].endswith(".tif"):
            continue
        name = Path(row["patient"]).stem
        patient = name.split("_node_")[0]
        slides.append(
            Slide(
                slide_id=name,
                path=root / "images" / row["patient"],
                patient_id=patient,
                label=CAMELYON17_CLASSES.index(row["stage"]),
                stratum=stages[patient],
            )
        )
    return slides


class CamelyonDataModule(SlideFeatureDataModule):
    """Fetches the label files and slides of one CAMELYON dataset from the public bucket."""

    dataset: ClassVar[str]
    labels: ClassVar[str]

    def __init__(self, download: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.download = download

    def prepare_data(self) -> None:
        if self.download:
            bucket.fetch_metadata(
                bucket.client(), self.dataset, self.root, (self.labels, bucket.CHECKSUMS)
            )
        super().prepare_data()

    def fetch_slide(self, slide: Slide) -> None:
        if not self.download:
            super().fetch_slide(slide)
            return
        name = slide.path.relative_to(self.root).as_posix()
        expected = bucket.read_checksums(self.root / bucket.CHECKSUMS).get(name)
        bucket.fetch_file(bucket.client(), self.dataset, name, slide.path, expected)


class Camelyon16DataModule(CamelyonDataModule):
    """Slide-level metastasis detection, ``normal`` vs ``tumor``, on the official split.

    CAMELYON16 publishes slide IDs only, so each slide counts as its own patient and the
    run records that patient-level leakage could not be checked.
    """

    dataset = "CAMELYON16"
    labels = "evaluation/reference.csv"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("split", "predefined")
        super().__init__(**kwargs)
        self.info = DatasetInfo(
            label_mode="binary",
            class_names=("normal", "tumor"),
            in_channels=3,
            patient_ids_available=False,
            license="CC0-1.0",
            citation="Ehteshami Bejnordi B. et al. JAMA 318(22):2199-2210 (2017). "
            "https://doi.org/10.1001/jama.2017.14585. " + DATASET_CITATION,
        )

    def slides(self) -> list[Slide]:
        return camelyon16_slides(self.root)


class Camelyon17DataModule(CamelyonDataModule):
    """Slide-level metastasis size, ``negative``/``itc``/``micro``/``macro``, on the 100
    labelled training patients with ``split: patient``. There is no official test split to
    use: the challenge never released labels for its test patients."""

    dataset = "CAMELYON17"
    labels = "stages.csv"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("split", "patient")
        super().__init__(**kwargs)
        self.info = DatasetInfo(
            label_mode="multiclass",
            class_names=CAMELYON17_CLASSES,
            in_channels=3,
            license="CC0-1.0",
            citation="Bandi P. et al. IEEE TMI 38(2):550-560 (2019). "
            "https://doi.org/10.1109/TMI.2018.2867350. " + DATASET_CITATION,
        )

    def slides(self) -> list[Slide]:
        return camelyon17_slides(self.root)
