"""Unit tests for Observable Behaviour Recognition (Feature 5).

Tests person crop preprocessing, behaviour label definitions, prototype heuristics,
PyTorch model loading/inference, frame annotation, and end-to-end pipeline execution.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.behaviour.behaviour_classifier import (
    BehaviourClassifier,
    BehaviourPrediction,
    BehaviourSummary,
    run_behaviour_recognition_on_tracks,
)
from src.behaviour.behaviour_labels import (
    ALL_BEHAVIOUR_CLASSES,
    BEHAVIOUR_DESCRIPTIONS,
    CLASS_HEAD_DOWN,
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_READING_WRITING,
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
    get_behaviour_bgr,
    get_behaviour_description,
    get_behaviour_hex,
    get_behaviour_rgb,
)
from src.behaviour.preprocessing import preprocess_person_crop
from src.config import BEHAVIOURS_CSV_FILENAME, DEFAULT_CROP_SIZE


def test_target_behaviour_labels_and_metadata():
    """Verify all six target observable classes and fallback unknown state are defined."""
    assert len(TARGET_BEHAVIOUR_CLASSES) == 6
    assert len(ALL_BEHAVIOUR_CLASSES) == 7
    assert CLASS_UNKNOWN in ALL_BEHAVIOUR_CLASSES

    # Verify each class has non-empty descriptions and valid colors
    for cls in ALL_BEHAVIOUR_CLASSES:
        desc = get_behaviour_description(cls)
        assert "observable_evidence" in desc and len(desc["observable_evidence"]) > 0
        assert "scientific_boundary" in desc and len(desc["scientific_boundary"]) > 0

        rgb = get_behaviour_rgb(cls)
        assert isinstance(rgb, tuple) and len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)

        bgr = get_behaviour_bgr(cls)
        assert bgr == (rgb[2], rgb[1], rgb[0])

        hex_code = get_behaviour_hex(cls)
        assert hex_code.startswith("#") and len(hex_code) == 7


def test_preprocess_person_crop_valid():
    """Verify valid person crop extraction, resizing, and PyTorch tensor normalization."""
    # Create 480x640 synthetic RGB image
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Put a textured pattern in the bounding box area
    cv2.circle(frame, (200, 200), 50, (220, 180, 140), -1)

    bbox = (150.0, 120.0, 260.0, 320.0)
    ok, crop_rgb, crop_tensor, err = preprocess_person_crop(frame, bbox, target_size=DEFAULT_CROP_SIZE)

    assert ok, f"Preprocessing failed: {err}"
    assert crop_rgb is not None
    assert crop_rgb.shape == (224, 224, 3)
    assert crop_rgb.dtype == np.uint8

    assert crop_tensor is not None
    assert isinstance(crop_tensor, torch.Tensor)
    assert crop_tensor.shape == (1, 3, 224, 224)
    assert crop_tensor.dtype == torch.float32


def test_preprocess_person_crop_boundary_clipping():
    """Verify bounding boxes that extend outside frame boundaries are safely clipped."""
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    # Box extending past left (<0) and bottom (>480)
    bbox = (-30.0, 100.0, 180.0, 520.0)
    ok, crop_rgb, crop_tensor, err = preprocess_person_crop(frame, bbox)

    assert ok
    assert crop_rgb is not None
    assert crop_rgb.shape == (224, 224, 3)


def test_preprocess_person_crop_invalid_inputs():
    """Verify graceful rejection of empty, zero-area, inverted, or tiny crops."""
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    # Inverted bbox
    ok1, _, _, err1 = preprocess_person_crop(frame, (200.0, 200.0, 100.0, 100.0))
    assert not ok1
    assert "Invalid bounding box geometry" in err1

    # Zero area bbox
    ok2, _, _, err2 = preprocess_person_crop(frame, (100.0, 100.0, 100.0, 100.0))
    assert not ok2

    # Sub-minimum crop
    ok3, _, _, err3 = preprocess_person_crop(frame, (100.0, 100.0, 105.0, 105.0))
    assert not ok3
    assert "too small" in err3

    # None frame
    ok4, _, _, err4 = preprocess_person_crop(None, (10.0, 10.0, 100.0, 100.0))
    assert not ok4
    assert "Invalid or empty input frame" in err4


def test_classifier_prototype_mode_initialization():
    """Verify classifier initializes in Prototype / Baseline mode when no weights exist."""
    classifier = BehaviourClassifier(weights_path=Path("non_existent_weights.pt"))
    assert not classifier.is_trained_model
    assert "Prototype" in classifier.mode_name


def test_classifier_prototype_prediction():
    """Verify prototype heuristic assigns valid behaviour class and confidence."""
    classifier = BehaviourClassifier()
    # Synthetic upright student crop with texture
    crop = np.full((224, 224, 3), 140, dtype=np.uint8)
    cv2.circle(crop, (112, 50), 30, (200, 160, 120), -1)  # Head
    cv2.rectangle(crop, (50, 150), (180, 210), (80, 80, 80), -1)  # Desk/hands

    track_info = {"track_id": 1, "x1": 100.0, "y1": 100.0, "x2": 250.0, "y2": 320.0}
    beh_class, conf, evidence = classifier.predict_person(
        crop_rgb=crop,
        crop_tensor=None,
        track_info=track_info,
        frame_tracks=[track_info],
        frame_dims=(720, 1280),
        conf_threshold=0.30,
    )

    assert beh_class in ALL_BEHAVIOUR_CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(evidence) > 0


def test_classifier_peer_interaction_heuristic():
    """Verify nearby adjacent peers trigger interacting with peers classification."""
    classifier = BehaviourClassifier()
    crop = np.full((224, 224, 3), 140, dtype=np.uint8)
    cv2.line(crop, (10, 10), (200, 200), (255, 255, 255), 3)

    track1 = {"track_id": 1, "x1": 200.0, "y1": 300.0, "x2": 350.0, "y2": 550.0}
    # Track 2 is immediately adjacent within conversational distance
    track2 = {"track_id": 2, "x1": 370.0, "y1": 310.0, "x2": 520.0, "y2": 560.0}

    beh_class, conf, evidence = classifier.predict_person(
        crop_rgb=crop,
        crop_tensor=None,
        track_info=track1,
        frame_tracks=[track1, track2],
        frame_dims=(720, 1280),
        conf_threshold=0.30,
    )

    assert beh_class == CLASS_INTERACTING_WITH_PEERS
    assert conf >= 0.50
    assert "peer" in evidence.lower()


def test_classifier_unknown_handling_on_blur():
    """Verify blurry or low-texture crops are assigned Unknown / Uncertain."""
    classifier = BehaviourClassifier()
    # Blank completely smooth crop (Laplacian variance == 0)
    smooth_crop = np.full((224, 224, 3), 128, dtype=np.uint8)

    track = {"track_id": 3, "x1": 100.0, "y1": 100.0, "x2": 250.0, "y2": 320.0}
    beh_class, conf, evidence = classifier.predict_person(
        crop_rgb=smooth_crop,
        crop_tensor=None,
        track_info=track,
        frame_tracks=[track],
        frame_dims=(720, 1280),
        conf_threshold=0.40,
    )

    assert beh_class == CLASS_UNKNOWN
    assert "texture" in evidence.lower() or "blur" in evidence.lower() or "below" in evidence.lower()


def test_classifier_draw_behaviours():
    """Verify visual annotation renders bounding box and multi-line badge."""
    classifier = BehaviourClassifier()
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    pred = BehaviourPrediction(
        video_id="test_vid",
        frame_id=0,
        extracted_frame_index=1,
        timestamp_seconds=0.0,
        frame_filename="frame_000001.jpg",
        track_id=4,
        behaviour_class=CLASS_READING_WRITING,
        confidence=0.84,
        x1=100.0,
        y1=100.0,
        x2=220.0,
        y2=300.0,
        visual_evidence="Downward gaze toward notebook with active edge texture.",
    )

    annotated = classifier.draw_behaviours(frame, [pred])
    assert annotated is not None
    assert annotated.shape == frame.shape
    # Frame should contain drawn pixels
    assert np.any(annotated != 40)


def test_classifier_pytorch_model_mock():
    """Verify Situation A: PyTorch model forward pass predicts class cleanly."""
    # Create simple mock PyTorch neural network
    class MockBehaviourNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(3, len(TARGET_BEHAVIOUR_CLASSES))

        def forward(self, x):
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    classifier = BehaviourClassifier()
    classifier.model = MockBehaviourNet()
    classifier.is_trained_model = True
    assert "PyTorch" in classifier.mode_name

    crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    _, _, tensor, _ = preprocess_person_crop(crop, (0.0, 0.0, 224.0, 224.0))

    track = {"track_id": 1, "x1": 50.0, "y1": 50.0, "x2": 150.0, "y2": 250.0}
    beh_class, conf, evidence = classifier.predict_person(
        crop_rgb=crop,
        crop_tensor=tensor,
        track_info=track,
        frame_tracks=[track],
        frame_dims=(480, 640),
        conf_threshold=0.0,  # Low threshold to accept mock prediction
    )

    assert beh_class in ALL_BEHAVIOUR_CLASSES
    assert 0.0 <= conf <= 1.0


def test_run_behaviour_recognition_pipeline(tmp_path):
    """Verify batch behaviour recognition pipeline, summary generation, and CSV export."""
    # Create 2 synthetic frame files
    frame_paths = []
    for i in range(2):
        p = tmp_path / f"frame_{i:06d}.jpg"
        img = np.full((360, 640, 3), 120, dtype=np.uint8)
        cv2.circle(img, (150, 150), 40, (220, 180, 140), -1)
        cv2.line(img, (50, 250), (250, 250), (255, 255, 255), 2)
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frame_paths.append(str(p))

    frames_df = pd.DataFrame({
        "frame_index": [0, 5],
        "extracted_frame_index": [1, 2],
        "timestamp_seconds": [0.0, 0.167],
        "frame_filename": ["frame_000000.jpg", "frame_000001.jpg"],
        "frame_path": frame_paths,
    })

    tracks_df = pd.DataFrame({
        "video_id": ["test_video", "test_video"],
        "frame_id": [0, 5],
        "extracted_frame_index": [1, 2],
        "timestamp_seconds": [0.0, 0.167],
        "frame_filename": ["frame_000000.jpg", "frame_000001.jpg"],
        "track_id": [1, 1],
        "class_id": [0, 0],
        "class_name": ["person", "person"],
        "confidence": [0.88, 0.86],
        "x1": [100.0, 102.0],
        "y1": [100.0, 101.0],
        "x2": [220.0, 222.0],
        "y2": [300.0, 301.0],
        "center_x": [160.0, 162.0],
        "center_y": [200.0, 201.0],
    })

    classifier = BehaviourClassifier()
    ok, summary, behaviours_df, annotated_frames, msg = run_behaviour_recognition_on_tracks(
        video_id="test_video_pipeline",
        frames_df=frames_df,
        tracks_df=tracks_df,
        classifier=classifier,
        conf_threshold=0.30,
        frame_selection_mode="all",
    )

    assert ok, f"Pipeline failed: {msg}"
    assert summary is not None
    assert isinstance(summary, BehaviourSummary)
    assert summary.total_observations == 2
    assert summary.unique_tracks == 1
    assert summary.behaviours_csv_path.exists()

    assert behaviours_df is not None
    expected_cols = {
        "video_id", "frame_id", "extracted_frame_index", "timestamp_seconds",
        "frame_filename", "track_id", "behaviour_class", "confidence",
        "x1", "y1", "x2", "y2", "visual_evidence"
    }
    assert expected_cols.issubset(set(behaviours_df.columns))

    # Verify CSV file contents
    loaded_csv = pd.read_csv(summary.behaviours_csv_path)
    assert len(loaded_csv) == 2
    assert expected_cols.issubset(set(loaded_csv.columns))


def test_run_behaviour_recognition_invalid_inputs():
    """Verify graceful handling when frames_df or tracks_df is empty."""
    classifier = BehaviourClassifier()
    empty_df = pd.DataFrame()

    ok1, _, _, _, msg1 = run_behaviour_recognition_on_tracks(
        video_id="test",
        frames_df=empty_df,
        tracks_df=pd.DataFrame({"track_id": [1]}),
        classifier=classifier,
    )
    assert not ok1
    assert "No extracted frames" in msg1

    ok2, _, _, _, msg2 = run_behaviour_recognition_on_tracks(
        video_id="test",
        frames_df=pd.DataFrame({"frame_index": [0]}),
        tracks_df=empty_df,
        classifier=classifier,
    )
    assert not ok2
    assert "No tracking data" in msg2
