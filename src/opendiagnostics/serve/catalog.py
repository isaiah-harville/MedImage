"""How each servable model is described to people: the scan it reads and what it answers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanType:
    scan_type: str
    modality: str
    question: str


MEDMNIST = {
    "pathmnist": ScanType(
        "Colon pathology slide", "Histology (H&E)", "Which tissue type is this patch?"
    ),
    "chestmnist": ScanType("Chest X-ray", "X-ray", "Which of 14 findings are present?"),
    "dermamnist": ScanType(
        "Skin lesion photo", "Dermatoscopy", "Which type of skin lesion is this?"
    ),
    "octmnist": ScanType(
        "Retinal OCT scan", "Optical coherence tomography", "Which retinal condition?"
    ),
    "pneumoniamnist": ScanType("Pediatric chest X-ray", "X-ray", "Pneumonia or normal?"),
    "retinamnist": ScanType(
        "Retinal fundus photo", "Fundus photography", "Which diabetic retinopathy grade (0 to 4)?"
    ),
    "breastmnist": ScanType("Breast ultrasound", "Ultrasound", "Malignant, or normal or benign?"),
    "bloodmnist": ScanType("Blood cell", "Microscopy", "Which blood cell type?"),
    "tissuemnist": ScanType("Kidney cortex cells", "Microscopy", "Which kidney cell type?"),
    "organamnist": ScanType("Abdominal CT, axial slice", "CT", "Which organ is in view?"),
    "organcmnist": ScanType("Abdominal CT, coronal slice", "CT", "Which organ is in view?"),
    "organsmnist": ScanType("Abdominal CT, sagittal slice", "CT", "Which organ is in view?"),
}

ORDER = list(MEDMNIST)


def describe(flag: str | None, model_id: str) -> ScanType:
    """The MedMNIST subset's description, or the model id for anything else."""
    return MEDMNIST.get(flag or "", ScanType(model_id, "unknown", "See the model card."))
