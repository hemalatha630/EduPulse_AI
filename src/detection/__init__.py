"""Student and person detection package for EduPulse AI."""

from src.detection.detector import (
    DetectionResult,
    DetectionSummary,
    YOLOPersonDetector,
    run_detection_on_frames,
)

__all__ = [
    "DetectionResult",
    "DetectionSummary",
    "YOLOPersonDetector",
    "run_detection_on_frames",
]
