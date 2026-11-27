"""How each servable model is described to people: the scan it reads, what it answers, and the
group of imaging types the demo site lists it under."""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = {
    "CT": "CT and MRI",
    "MRI": "CT and MRI",
    "X-ray": "X-ray and mammography",
    "Mammography": "X-ray and mammography",
    "Ultrasound": "Ultrasound",
    "Histology (H&E)": "Pathology and microscopy",
    "Microscopy": "Pathology and microscopy",
    "Dermatoscopy": "Skin",
    "Optical coherence tomography": "Eye",
    "Fundus photography": "Eye",
}
ORDER = [*dict.fromkeys(CATEGORIES.values()), "Other"]


@dataclass(frozen=True)
class ScanType:
    scan_type: str
    modality: str
    question: str

    @property
    def category(self) -> str:
        return CATEGORIES.get(self.modality, "Other")


DATASETS = {
    "organamnist": ScanType("Abdominal CT, axial slice", "CT", "Which organ is in view?"),
    "organcmnist": ScanType("Abdominal CT, coronal slice", "CT", "Which organ is in view?"),
    "organsmnist": ScanType("Abdominal CT, sagittal slice", "CT", "Which organ is in view?"),
    "chestmnist": ScanType("Chest X-ray", "X-ray", "Which of 14 findings are present?"),
    "nih-chestxray14": ScanType("Chest X-ray", "X-ray", "Which of 14 findings are present?"),
    "pneumoniamnist": ScanType("Pediatric chest X-ray", "X-ray", "Pneumonia or normal?"),
    "kermany-cxr": ScanType("Pediatric chest X-ray", "X-ray", "Pneumonia or normal?"),
    "cmmd": ScanType("Mammogram", "Mammography", "Malignant or benign?"),
    "breastmnist": ScanType("Breast ultrasound", "Ultrasound", "Malignant, or normal or benign?"),
    "pathmnist": ScanType(
        "Colon pathology slide", "Histology (H&E)", "Which tissue type is this patch?"
    ),
    "pcam": ScanType(
        "Lymph node slide patch", "Histology (H&E)", "Does the patch contain metastatic tumor?"
    ),
    "bloodmnist": ScanType("Blood cell", "Microscopy", "Which blood cell type?"),
    "tissuemnist": ScanType("Kidney cortex cells", "Microscopy", "Which kidney cell type?"),
    "dermamnist": ScanType(
        "Skin lesion photo", "Dermatoscopy", "Which type of skin lesion is this?"
    ),
    "ham10000": ScanType("Skin lesion photo", "Dermatoscopy", "Which type of skin lesion is this?"),
    "isic2019": ScanType("Skin lesion photo", "Dermatoscopy", "Which type of skin lesion is this?"),
    "isic2020": ScanType("Skin lesion photo", "Dermatoscopy", "Melanoma or not?"),
    "octmnist": ScanType(
        "Retinal OCT scan", "Optical coherence tomography", "Which retinal condition?"
    ),
    "kermany-oct": ScanType(
        "Retinal OCT scan", "Optical coherence tomography", "Which retinal condition?"
    ),
    "retinamnist": ScanType(
        "Retinal fundus photo", "Fundus photography", "Which diabetic retinopathy grade (0 to 4)?"
    ),
}


def dataset_of(model_id: str) -> str:
    """The dataset part of a project name: ``kermany-oct-convnext`` reads ``kermany-oct``."""
    return model_id.rsplit("-", 1)[0]


def describe(model_id: str) -> ScanType:
    """The dataset's description, or the model id for a dataset not described here."""
    return DATASETS.get(dataset_of(model_id), ScanType(model_id, "unknown", "See the model card."))


def rank(model_id: str) -> tuple[int, int, str]:
    """Sort key that lists models by category, then in the order of ``DATASETS``."""
    datasets = list(DATASETS)
    dataset = dataset_of(model_id)
    position = datasets.index(dataset) if dataset in datasets else len(datasets)
    return (ORDER.index(describe(model_id).category), position, model_id)
