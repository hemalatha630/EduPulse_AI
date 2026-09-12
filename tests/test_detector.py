"""Unit and integration tests for Feature 3: Student / Person Detection."""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest

from src.detection.detector import (
    DetectionResult,
    DetectionSummary,
    YOLOPersonDetector,
    run_detection_on_frames,
)


@pytest.fixture(scope="module")
def real_classroom_image() -> np.ndarray:
    """Load or generate a realistic classroom image containing people."""
    img_candidates = list(
        Path(r"C:\Users\hemalatha\.gemini\antigravity-ide\brain\b8e18624-882a-4b54-a1f2-7c06b39365f5").glob(
            "classroom_students_real*.jpg"
        )
    )
    if img_candidates and img_candidates[0].exists():
        bgr = cv2.imread(str(img_candidates[0]))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Fallback to extracted frame if exists
    extracted_sample = Path("data/frames/classroom_lecture_demo/frame_000001.jpg")
    if extracted_sample.exists():
        bgr = cv2.imread(str(extracted_sample))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Synthetic fallback with basic person-like shapes
    synthetic = np.full((720, 1280, 3), 220, dtype=np.uint8)
    return synthetic


@pytest.fixture(scope="module")
def detector() -> YOLOPersonDetector:
    """Initialize and load YOLO detector once for tests."""
    det = YOLOPersonDetector()
    ok, msg = det.load_model()
    assert ok is True, f"Detector failed to load: {msg}"
    return det


# =====================================================================
# Tests: Model Loading & Initialization
# =====================================================================


def test_detector_initialization(detector):
    """Test 1: Verify YOLO detector initializes and model weights exist."""
    assert detector._model is not None
    assert detector.model_name == "yolov8n.pt"
    assert detector.model_path.exists()


# =====================================================================
# Tests: Person Detection on Classroom Frame
# =====================================================================


def test_detector_detect_on_classroom(detector, real_classroom_image):
    """Test 2: Verify person detection identifies people with valid bboxes and confidence."""
    ok, detections, msg = detector.detect(real_classroom_image, conf_threshold=0.25)
    assert ok is True
    assert msg == ""
    assert len(detections) > 0, "Expected at least one person detected in classroom image"

    for det in detections:
        assert det.class_id == 0
        assert det.class_name == "person"
        assert 0.25 <= det.confidence <= 1.0
        assert 0 <= det.x1 < det.x2 <= real_classroom_image.shape[1] + 5
        assert 0 <= det.y1 < det.y2 <= real_classroom_image.shape[0] + 5
        assert det.width > 0
        assert det.height > 0


# =====================================================================
# Tests: Confidence Threshold Filtering
# =====================================================================


def test_detector_confidence_threshold_filtering(detector, real_classroom_image):
    """Test 3: Verify higher confidence threshold produces fewer or equal detections."""
    ok_low, dets_low, _ = detector.detect(real_classroom_image, conf_threshold=0.20)
    ok_high, dets_high, _ = detector.detect(real_classroom_image, conf_threshold=0.75)

    assert ok_low is True
    assert ok_high is True
    assert len(dets_high) <= len(dets_low)
    for d in dets_high:
        assert d.confidence >= 0.75


# =====================================================================
# Tests: Bounding Box Visualization
# =====================================================================


def test_detector_draw_detections(detector, real_classroom_image):
    """Test 4: Verify bounding box rendering on frame."""
    ok, detections, _ = detector.detect(real_classroom_image, conf_threshold=0.25)
    assert ok is True

    annotated = detector.draw_detections(real_classroom_image, detections)
    assert annotated is not None
    assert annotated.shape == real_classroom_image.shape
    assert annotated.dtype == np.uint8
    # Image should be modified if detections were drawn
    if len(detections) > 0:
        assert not np.array_equal(annotated, real_classroom_image)


# =====================================================================
# Tests: Detection Pipeline & CSV Export
# =====================================================================


def test_run_detection_on_frames_pipeline(detector):
    """Test 5: Verify multi-frame detection pipeline and CSV schema."""
    metadata_csv = Path("data/processed/classroom_lecture_demo/frame_metadata.csv")
    if not metadata_csv.exists():
        pytest.skip("classroom_lecture_demo metadata CSV not found.")

    frames_df = pd.read_csv(metadata_csv)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ok, summary, det_df, annotated, msg = run_detection_on_frames(
            video_id="classroom_lecture_demo",
            frames_df=frames_df,
            detector=detector,
            conf_threshold=0.25,
            frame_selection_mode="sample",
            processed_base_dir=tmp_path,
        )

        assert ok is True
        assert summary is not None
        assert summary.frames_processed == 3
        assert summary.total_person_detections > 0
        assert summary.avg_detections_per_frame > 0
        assert summary.detections_csv_path.exists()

        # Check CSV schema
        csv_df = pd.read_csv(summary.detections_csv_path)
        expected_cols = [
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
        for col in expected_cols:
            assert col in csv_df.columns

        # Verify no tracking IDs exist
        assert "track_id" not in csv_df.columns
        assert "student_id" not in csv_df.columns


# =====================================================================
# Tests: Zero-Person Frame Handling
# =====================================================================


def test_detector_zero_person_frame(detector):
    """Test 6: Verify blank frame handles zero detections gracefully."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    ok, detections, msg = detector.detect(blank, conf_threshold=0.50)

    assert ok is True
    assert len(detections) == 0
    assert msg == ""

    # Drawing 0 detections should return original unchanged
    annotated = detector.draw_detections(blank, detections)
    assert np.array_equal(annotated, blank)


# =====================================================================
# Tests: Validation & Error Handling
# =====================================================================


def test_detector_invalid_inputs(detector):
    """Test 7: Verify invalid inputs are handled safely."""
    # None frame
    ok, dets, msg = detector.detect(None)
    assert ok is False
    assert len(dets) == 0

    # Invalid threshold (> 1.0)
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, dets, msg = detector.detect(blank, conf_threshold=1.5)
    assert ok is False
