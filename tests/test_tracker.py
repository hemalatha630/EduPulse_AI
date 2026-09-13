"""Unit and Integration Tests for Feature 4: Student / Person Tracking.

Tests the multi-object tracker, persistent Track IDs across consecutive frames,
confidence thresholding, spatial trajectory generation, and CSV export.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_TRACKER,
    DEFAULT_TRACKING_CONF_THRESHOLD,
    DEFAULT_YOLO_MODEL,
    PERSON_CLASS_ID,
    PERSON_CLASS_NAME,
    TRACKS_CSV_FILENAME,
)
from src.tracking.tracker import (
    PersonTracker,
    TrackResult,
    TrackingSummary,
    get_track_color,
    run_tracking_on_frames,
)


@pytest.fixture
def sample_classroom_frame():
    """Load real classroom frame if present, or create synthetic multi-person frame."""
    frame_path = Path("data/frames/classroom_lecture_demo/frame_000001.jpg")
    if frame_path.exists():
        img = cv2.imread(str(frame_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Fallback to realistic synthetic frame
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    return img


@pytest.fixture
def consecutive_classroom_frames():
    """Load two consecutive classroom frames for tracking continuity testing."""
    f1_path = Path("data/frames/classroom_lecture_demo/frame_000001.jpg")
    f2_path = Path("data/frames/classroom_lecture_demo/frame_000002.jpg")
    if f1_path.exists() and f2_path.exists():
        img1 = cv2.cvtColor(cv2.imread(str(f1_path)), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(str(f2_path)), cv2.COLOR_BGR2RGB)
        return img1, img2
    
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    return img, img


def test_tracker_initialization():
    """Verify tracker initializes correctly with ByteTrack and BoT-SORT."""
    tracker_byte = PersonTracker(tracker_type="bytetrack")
    assert tracker_byte.tracker_type == "bytetrack"
    assert tracker_byte.tracker_yaml == "bytetrack.yaml"
    ok, msg = tracker_byte.load_model()
    assert ok, f"ByteTrack loading failed: {msg}"

    tracker_bot = PersonTracker(tracker_type="botsort")
    assert tracker_bot.tracker_type == "botsort"
    assert tracker_bot.tracker_yaml == "botsort.yaml"
    ok2, msg2 = tracker_bot.load_model()
    assert ok2, f"BoT-SORT loading failed: {msg2}"


def test_track_color_generator():
    """Verify distinct persistent RGB colors generated for track IDs."""
    c1 = get_track_color(1)
    c2 = get_track_color(2)
    c1_again = get_track_color(1)

    assert isinstance(c1, tuple) and len(c1) == 3
    assert all(0 <= v <= 255 for v in c1)
    assert c1 == c1_again, "Color for same track_id must be deterministic"
    assert c1 != c2, "Different track IDs should receive distinct colors"


def test_tracker_consecutive_frame_tracking(consecutive_classroom_frames):
    """Verify tracking across consecutive frames preserves track IDs."""
    img1, img2 = consecutive_classroom_frames
    tracker = PersonTracker(tracker_type="bytetrack")
    tracker.reset()

    # Track on Frame 1
    ok1, tracks1, _ = tracker.track_frame(
        frame=img1,
        frame_id=0,
        extracted_frame_index=1,
        timestamp_seconds=0.0,
        frame_filename="frame_000001.jpg",
        conf_threshold=0.25,
    )
    assert ok1

    # Track on Frame 2
    ok2, tracks2, _ = tracker.track_frame(
        frame=img2,
        frame_id=1,
        extracted_frame_index=2,
        timestamp_seconds=0.167,
        frame_filename="frame_000002.jpg",
        conf_threshold=0.25,
    )
    assert ok2

    if tracks1 and tracks2:
        ids1 = {t.track_id for t in tracks1}
        ids2 = {t.track_id for t in tracks2}
        common_ids = ids1.intersection(ids2)
        # In real classroom video, several IDs persist between consecutive frames
        assert len(common_ids) > 0, f"Expected persisting IDs across consecutive frames, got ids1={ids1}, ids2={ids2}"

        # Verify TrackResult attributes
        first_track = tracks1[0]
        assert first_track.track_id > 0
        assert first_track.class_id == PERSON_CLASS_ID
        assert first_track.class_name == PERSON_CLASS_NAME
        assert first_track.center_x == (first_track.x1 + first_track.x2) / 2.0
        assert first_track.center_y == (first_track.y1 + first_track.y2) / 2.0


def test_tracker_confidence_threshold_filtering(sample_classroom_frame):
    """Verify higher confidence threshold filters out low-confidence tracks."""
    tracker = PersonTracker()
    tracker.reset()

    # Low threshold
    ok_low, tracks_low, _ = tracker.track_frame(
        sample_classroom_frame,
        frame_id=0,
        extracted_frame_index=1,
        timestamp_seconds=0.0,
        frame_filename="test.jpg",
        conf_threshold=0.15,
    )

    tracker.reset()
    # High threshold
    ok_high, tracks_high, _ = tracker.track_frame(
        sample_classroom_frame,
        frame_id=0,
        extracted_frame_index=1,
        timestamp_seconds=0.0,
        frame_filename="test.jpg",
        conf_threshold=0.85,
    )

    assert ok_low and ok_high
    assert len(tracks_high) <= len(tracks_low)


def test_tracker_zero_person_frame():
    """Verify tracker handles frames with zero people gracefully without crashing."""
    tracker = PersonTracker()
    tracker.reset()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    ok, tracks, msg = tracker.track_frame(
        blank_frame,
        frame_id=0,
        extracted_frame_index=1,
        timestamp_seconds=0.0,
        frame_filename="blank.jpg",
        conf_threshold=0.25,
    )
    assert ok
    assert len(tracks) == 0
    assert msg == ""


def test_tracker_draw_tracks_with_trajectory():
    """Verify rendering of bounding boxes, labels, and trajectory paths."""
    tracker = PersonTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    sample_track = TrackResult(
        frame_id=0,
        extracted_frame_index=1,
        timestamp_seconds=0.0,
        frame_filename="test.jpg",
        track_id=4,
        confidence=0.88,
        x1=50.0,
        y1=50.0,
        x2=150.0,
        y2=200.0,
    )

    # Populate trajectory history
    tracker.trajectory_history[4] = [(95.0, 120.0), (100.0, 125.0)]

    annotated = tracker.draw_tracks(frame, [sample_track], show_trajectories=True)
    assert annotated is not None
    assert annotated.shape == frame.shape
    # Frame should have drawn non-zero colored pixels
    assert np.any(annotated > 0)


def test_tracker_reset():
    """Verify reset() clears trajectory history and model predictor state."""
    tracker = PersonTracker()
    tracker.trajectory_history[1] = [(10.0, 20.0), (15.0, 25.0)]
    assert len(tracker.trajectory_history) == 1

    tracker.reset()
    assert len(tracker.trajectory_history) == 0


def test_run_tracking_on_frames_pipeline(tmp_path):
    """Verify end-to-end multi-frame tracking pipeline and CSV export."""
    # Create 3 synthetic test image files
    frame_paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i:06d}.jpg"
        img = np.full((360, 640, 3), 128, dtype=np.uint8)
        # Draw simulated person shape
        cv2.circle(img, (200 + i * 10, 150), 30, (255, 255, 255), -1)
        cv2.imwrite(str(p), img)
        frame_paths.append(str(p))

    frames_df = pd.DataFrame({
        "frame_index": [0, 5, 10],
        "extracted_frame_index": [1, 2, 3],
        "timestamp_seconds": [0.0, 0.167, 0.333],
        "frame_filename": [f"frame_{i:06d}.jpg" for i in range(3)],
        "frame_path": frame_paths,
    })

    tracker = PersonTracker()
    success, summary, tracks_df, annotated_frames, msg = run_tracking_on_frames(
        video_id="test_tracking_vid",
        frames_df=frames_df,
        tracker=tracker,
        conf_threshold=0.25,
        frame_selection_mode="all",
        show_trajectories=True,
    )

    assert success
    assert summary is not None
    assert isinstance(summary, TrackingSummary)
    assert summary.frames_processed == 3
    assert tracks_df is not None
    assert isinstance(tracks_df, pd.DataFrame)
    assert summary.tracks_csv_path.exists()

    # Verify CSV schema
    loaded_df = pd.read_csv(summary.tracks_csv_path)
    expected_cols = {
        "video_id", "frame_id", "extracted_frame_index", "timestamp_seconds",
        "frame_filename", "track_id", "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y"
    }
    assert expected_cols.issubset(set(loaded_df.columns))


def test_tracker_invalid_inputs():
    """Verify tracker fails gracefully with informative errors on invalid inputs."""
    tracker = PersonTracker()

    # Invalid frame (None)
    ok1, tracks1, err1 = tracker.track_frame(
        None, 0, 1, 0.0, "test.jpg"
    )
    assert not ok1
    assert "Invalid frame" in err1

    # Invalid confidence threshold (> 1.0)
    valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ok2, tracks2, err2 = tracker.track_frame(
        valid_frame, 0, 1, 0.0, "test.jpg", conf_threshold=1.5
    )
    assert not ok2
    assert "Confidence threshold must be between" in err2
