from opendiagnostics.modules.base import DiagnosticModule, Prediction
from opendiagnostics.modules.classification import ImageClassifier
from opendiagnostics.modules.detection import ObjectDetector
from opendiagnostics.modules.detection_3d import VolumeDetector
from opendiagnostics.modules.segmentation import SegmentationModule
from opendiagnostics.modules.slide import SlideClassifier

__all__ = [
    "DiagnosticModule",
    "ImageClassifier",
    "ObjectDetector",
    "Prediction",
    "SegmentationModule",
    "SlideClassifier",
    "VolumeDetector",
]
