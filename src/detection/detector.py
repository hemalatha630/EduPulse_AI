"""Student / Person Detection Module for EduPulse AI.

Implements lightweight pretrained YOLO-based person detection in classroom video frames,
computes bounding box coordinates, and generates structured detection metadata without
assigning persistent tracking IDs.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_YOLO_MODEL,
    MODELS_DIR,
    PERSON_CLASS_ID,
    PERSON_CLASS_NAME,
    PROCESSED_DIR,
)
from src.preprocessing.frame_preprocessor import FramePreprocessor


@dataclass
class DetectionResult:
    """Individual person detection within a single frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = PERSON_CLASS_ID
    class_name: str = PERSON_CLASS_NAME
    detection_index: int = 1

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        """Convert detection result to serializable dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "detection_index": self.detection_index,
        }


@dataclass
class DetectionSummary:
    """Summary of person detections across processed frames."""

    video_id: str
    model_name: str
    confidence_threshold: float
    frames_processed: int
    total_person_detections: int
    avg_detections_per_frame: float
    min_detections: int
    max_detections: int
    detections_csv_path: Path
    status: str

    def to_dict(self) -> dict:
        """Convert summary to dictionary."""
        data = asdict(self)
        data["detections_csv_path"] = str(self.detections_csv_path)
        return data


class YOLOPersonDetector:
    """Lightweight YOLO-based object detector filtered strictly for the 'person' class."""

    def __init__(self, model_name: str = DEFAULT_YOLO_MODEL, model_dir: Path | None = None):
        """Initialize detector with specified YOLO model."""
        self.model_name = model_name
        self.model_dir = model_dir or MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / self.model_name
        self._model = None

    def load_model(self) -> Tuple[bool, str]:
        """Load pretrained YOLO weights, downloading automatically if not cached."""
        try:
            from ultralytics import YOLO

            # If model exists locally in models/ use it, otherwise download to models/
            target_path = self.model_path if self.model_path.exists() else self.model_name
            self._model = YOLO(str(target_path))

            # Ensure downloaded model is saved in models/ directory
            downloaded_weights = Path(f"{self.model_name}")
            if downloaded_weights.exists() and not self.model_path.exists():
                import shutil
                shutil.move(str(downloaded_weights), str(self.model_path))

            return True, f"Successfully loaded {self.model_name}"
        except Exception as exc:
            return False, f"Failed to load YOLO model: {str(exc)}"

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Tuple[bool, List[DetectionResult], str]:
        """Detect visible people in a single image frame.

        Args:
            frame: Numpy array representing image (RGB or BGR).
            conf_threshold: Minimum confidence threshold (0.0 to 1.0).

        Returns:
            Tuple of (success: bool, detections: List[DetectionResult], message: str)
        """
        if self._model is None:
            ok, err = self.load_model()
            if not ok:
                return False, [], err

        is_valid, val_err = FramePreprocessor.validate_frame(frame)
        if not is_valid:
            return False, [], f"Invalid frame passed to detector: {val_err}"

        if not (0.0 <= conf_threshold <= 1.0):
            return False, [], f"Confidence threshold must be between 0.0 and 1.0, got: {conf_threshold}"

        try:
            # Run inference targeting only class 0 ('person')
            results = self._model(
                frame,
                conf=conf_threshold,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )

            detections: List[DetectionResult] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for idx, box in enumerate(boxes, 1):
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        # Ensure class is person
                        if cls_id == PERSON_CLASS_ID and conf >= conf_threshold:
                            detections.append(
                                DetectionResult(
                                    x1=float(coords[0]),
                                    y1=float(coords[1]),
                                    x2=float(coords[2]),
                                    y2=float(coords[3]),
                                    confidence=conf,
                                    class_id=cls_id,
                                    class_name=PERSON_CLASS_NAME,
                                    detection_index=idx,
                                )
                            )

            return True, detections, ""
        except Exception as exc:
            return False, [], f"Inference error during person detection: {str(exc)}"

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[DetectionResult],
        box_color: Tuple[int, int, int] = (0, 210, 90),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Draw bounding boxes and confidence tags on a frame.

        Args:
            frame: Input image array (RGB format).
            detections: List of DetectionResult objects.
            box_color: Bounding box RGB color tuple.
            text_color: Text RGB color tuple.

        Returns:
            Annotated frame array in RGB format.
        """
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()

        for det in detections:
            x1, y1 = int(round(det.x1)), int(round(det.y1))
            x2, y2 = int(round(det.x2)), int(round(det.y2))

            # Clamp coordinates to frame boundaries
            h, w = annotated.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Draw outer bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Format label with temporary per-frame index
            label = f"Person #{det.detection_index} ({det.confidence:.2f})"

            # Calculate label background box size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw label background rectangle
            label_y1 = max(0, y1 - text_h - 8)
            label_y2 = y1
            label_x2 = min(w - 1, x1 + text_w + 6)
            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), box_color, -1)

            # Draw label text
            text_origin_y = y1 - 4 if y1 - text_h - 8 >= 0 else y1 + text_h + 4
            cv2.putText(
                annotated,
                label,
                (x1 + 3, text_origin_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated


def run_detection_on_frames(
    video_id: str,
    frames_df: pd.DataFrame,
    detector: YOLOPersonDetector,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    frame_selection_mode: str = "sample",
    custom_sample_count: int = 5,
    processed_base_dir: Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Tuple[bool, DetectionSummary | None, pd.DataFrame | None, Dict[int, np.ndarray], str]:
    """Execute person detection across selected frames from Feature 2.

    Args:
        video_id: Safe video identifier.
        frames_df: DataFrame of extracted frames from Feature 2 metadata CSV.
        detector: Initialized YOLOPersonDetector instance.
        conf_threshold: Confidence threshold for filtering detections.
        frame_selection_mode: 'sample' (first, mid, last), 'first', 'all', or 'custom'.
        custom_sample_count: Number of frames if custom selection is chosen.
        processed_base_dir: Destination base directory (defaults to PROCESSED_DIR).
        progress_callback: Optional callback receiving (current, total, msg).

    Returns:
        Tuple of (success: bool, summary: DetectionSummary or None, df: DataFrame or None,
                  annotated_frames: Dict[int, np.ndarray], message: str)
    """
    if frames_df is None or frames_df.empty:
        return False, None, None, {}, "No extracted frames provided for person detection."

    # Filter frame subset based on selection mode
    total_available = len(frames_df)
    if frame_selection_mode == "first":
        selected_df = frames_df.iloc[[0]].copy()
    elif frame_selection_mode == "all":
        selected_df = frames_df.copy()
    elif frame_selection_mode == "custom":
        step = max(1, total_available // max(1, custom_sample_count))
        indices = list(range(0, total_available, step))[:custom_sample_count]
        selected_df = frames_df.iloc[indices].copy()
    else:
        # Default: 'sample' (First, Middle, Last)
        if total_available == 1:
            indices = [0]
        elif total_available == 2:
            indices = [0, 1]
        else:
            indices = [0, total_available // 2, total_available - 1]
        selected_df = frames_df.iloc[indices].copy()

    processed_dir = (processed_base_dir or PROCESSED_DIR) / video_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    detections_csv_path = processed_dir / "detections.csv"

    detections_records: List[dict] = []
    annotated_frames: Dict[int, np.ndarray] = {}
    counts_per_frame: List[int] = []

    total_to_process = len(selected_df)

    for idx, (_, row) in enumerate(selected_df.iterrows(), 1):
        frame_path = Path(row["frame_path"])
        extracted_idx = int(row.get("extracted_frame_index", idx))
        timestamp_sec = float(row.get("timestamp_seconds", 0.0))

        if progress_callback:
            progress_callback(
                idx,
                total_to_process,
                f"Detecting people in frame {idx}/{total_to_process} (Frame #{extracted_idx})...",
            )

        if not frame_path.exists():
            continue

        read_ok, bgr_img, err = FramePreprocessor.load_frame(frame_path)
        if not read_ok or bgr_img is None:
            continue

        rgb_img = FramePreprocessor.bgr_to_rgb(bgr_img)
        det_ok, detections, det_err = detector.detect(rgb_img, conf_threshold=conf_threshold)

        if not det_ok:
            return False, None, None, {}, f"Detection failed on frame {extracted_idx}: {det_err}"

        # Draw bounding boxes on RGB image
        annotated_rgb = detector.draw_detections(rgb_img, detections)
        annotated_frames[extracted_idx] = annotated_rgb
        counts_per_frame.append(len(detections))

        # Record each detection for structured CSV output
        for det in detections:
            detections_records.append(
                {
                    "video_id": video_id,
                    "frame_id": extracted_idx,
                    "timestamp_seconds": timestamp_sec,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": round(float(det.confidence), 4),
                    "x1": round(float(det.x1), 2),
                    "y1": round(float(det.y1), 2),
                    "x2": round(float(det.x2), 2),
                    "y2": round(float(det.y2), 2),
                }
            )

    # Convert to DataFrame
    det_df = pd.DataFrame(detections_records)
    if det_df.empty:
        # Create empty DataFrame with required schema
        det_df = pd.DataFrame(
            columns=[
                "video_id",
                "frame_id",
                "timestamp_seconds",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

    # Save to CSV
    try:
        det_df.to_csv(detections_csv_path, index=False)
    except Exception as exc:
        return False, None, None, {}, f"Failed to save detections CSV: {str(exc)}"

    frames_processed_count = len(selected_df)
    total_detections_count = len(detections_records)
    avg_detections = round(total_detections_count / max(frames_processed_count, 1), 2)
    min_det = min(counts_per_frame) if counts_per_frame else 0
    max_det = max(counts_per_frame) if counts_per_frame else 0

    summary = DetectionSummary(
        video_id=video_id,
        model_name=detector.model_name,
        confidence_threshold=conf_threshold,
        frames_processed=frames_processed_count,
        total_person_detections=total_detections_count,
        avg_detections_per_frame=avg_detections,
        min_detections=min_det,
        max_detections=max_det,
        detections_csv_path=detections_csv_path,
        status="Completed",
    )

    if progress_callback:
        progress_callback(total_to_process, total_to_process, "Person detection complete!")

    return True, summary, det_df, annotated_frames, "Person detection completed successfully."
