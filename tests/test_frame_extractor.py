"""Unit and integration tests for Feature 2: Frame Extraction & Preprocessing."""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.video.frame_extractor import (
    ExtractionConfig,
    calculate_sampling_info,
    derive_video_id,
    extract_video_frames,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def test_video(temp_dir) -> Path:
    """Create a 30-frame synthetic video at 30 FPS (1.0 second duration, 320x240)."""
    video_path = temp_dir / "test_classroom.mp4"
    width, height = 320, 240
    fps = 30.0
    num_frames = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)
        # Add frame index text and dynamic shape
        cv2.putText(frame, f"F:{i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.circle(frame, (50 + i * 5, 150), 15, (255, 0, 0), -1)
        out.write(frame)

    out.release()
    return video_path


# =====================================================================
# Tests: Frame Preprocessor
# =====================================================================


def test_preprocessor_validate_frame_valid():
    valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    is_valid, err = FramePreprocessor.validate_frame(valid_frame)
    assert is_valid is True
    assert err == ""


def test_preprocessor_validate_frame_invalid():
    # None
    is_valid, err = FramePreprocessor.validate_frame(None)
    assert is_valid is False

    # Empty
    empty_frame = np.array([], dtype=np.uint8)
    is_valid, err = FramePreprocessor.validate_frame(empty_frame)
    assert is_valid is False

    # Wrong dimensions (2D without channels)
    frame_2d = np.zeros((100, 100), dtype=np.uint8)
    is_valid, err = FramePreprocessor.validate_frame(frame_2d)
    assert is_valid is False


def test_preprocessor_resizing():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)

    # Resize down
    resized = FramePreprocessor.resize_frame(frame, (200, 100))
    assert resized.shape == (100, 200, 3)

    # None target_size keeps original
    same = FramePreprocessor.resize_frame(frame, None)
    assert same.shape == (200, 400, 3)

    # Invalid target_size raises ValueError
    with pytest.raises(ValueError):
        FramePreprocessor.resize_frame(frame, (-10, 50))


def test_preprocessor_color_conversions():
    # Create pure blue in BGR: [255, 0, 0]
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[:, :] = [255, 0, 0]

    # Convert to RGB -> red channel should be 0, blue channel should be 255: [0, 0, 255]
    rgb = FramePreprocessor.bgr_to_rgb(bgr)
    assert np.array_equal(rgb[0, 0], [0, 0, 255])

    # Convert back to BGR
    back_to_bgr = FramePreprocessor.rgb_to_bgr(rgb)
    assert np.array_equal(back_to_bgr[0, 0], [255, 0, 0])


def test_preprocessor_save_and_load(temp_dir):
    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    save_path = temp_dir / "saved_frame.jpg"

    success, err = FramePreprocessor.save_frame(frame, save_path, quality=90)
    assert success is True
    assert save_path.exists()
    assert save_path.stat().st_size > 0

    load_success, loaded_frame, load_err = FramePreprocessor.load_frame(save_path)
    assert load_success is True
    assert loaded_frame is not None
    assert loaded_frame.shape == (120, 160, 3)


# =====================================================================
# Tests: Frame Extraction & Sampling
# =====================================================================


def test_calculate_sampling_info():
    eff_fps, desc = calculate_sampling_info(30.0, 1)
    assert eff_fps == 30.0
    assert "Every frame" in desc

    eff_fps, desc = calculate_sampling_info(30.0, 5)
    assert eff_fps == 6.0
    assert "Every 5th" in desc

    eff_fps, desc = calculate_sampling_info(29.97, 10)
    assert round(eff_fps, 2) == 3.0

    with pytest.raises(ValueError):
        calculate_sampling_info(30.0, 0)


def test_extract_video_frames_full_sampling(test_video, temp_dir):
    """Test 1: Extract every frame (sampling_interval=1)."""
    frames_dir = temp_dir / "frames"
    processed_dir = temp_dir / "processed"
    config = ExtractionConfig(sampling_interval=1)

    success, summary, df, msg = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=config,
    )

    assert success is True
    assert summary is not None
    assert df is not None
    assert summary.extracted_frames_count == 30
    assert len(df) == 30

    # Verify frame files exist on disk
    expected_video_id = derive_video_id(test_video)
    saved_files = list((frames_dir / expected_video_id).glob("*.jpg"))
    assert len(saved_files) == 30
    assert (frames_dir / expected_video_id / "frame_000001.jpg").exists()
    assert (frames_dir / expected_video_id / "frame_000030.jpg").exists()


def test_extract_video_frames_subsampled(test_video, temp_dir):
    """Test 2: Test different sampling rates (every 5th frame and every 10th frame)."""
    frames_dir = temp_dir / "frames"
    processed_dir = temp_dir / "processed"

    # 1. Every 5th frame on a 30-frame video -> exactly 6 frames (0, 5, 10, 15, 20, 25)
    cfg_5 = ExtractionConfig(sampling_interval=5, force_reextract=True)
    success, summary, df, msg = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=cfg_5,
    )
    assert success is True
    assert summary.extracted_frames_count == 6
    assert summary.effective_fps == 6.0
    assert len(df) == 6
    assert df["frame_index"].tolist() == [0, 5, 10, 15, 20, 25]

    # 2. Every 10th frame on a 30-frame video -> exactly 3 frames (0, 10, 20)
    cfg_10 = ExtractionConfig(sampling_interval=10, force_reextract=True)
    success, summary, df, msg = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=cfg_10,
    )
    assert success is True
    assert summary.extracted_frames_count == 3
    assert summary.effective_fps == 3.0
    assert len(df) == 3
    assert df["frame_index"].tolist() == [0, 10, 20]


def test_extract_frame_ordering_and_timestamps(test_video, temp_dir):
    """Test 3: Verify chronological timestamps and sequential frame numbers."""
    frames_dir = temp_dir / "frames"
    processed_dir = temp_dir / "processed"
    config = ExtractionConfig(sampling_interval=5)

    success, summary, df, msg = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=config,
    )
    assert success is True

    # Check strictly increasing chronological order
    timestamps = df["timestamp_seconds"].tolist()
    assert all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))
    assert timestamps[0] == 0.0

    # Check extracted frame indices are sequential starting from 1
    extracted_indices = df["extracted_frame_index"].tolist()
    assert extracted_indices == list(range(1, len(df) + 1))


def test_extract_metadata_csv(test_video, temp_dir):
    """Test 4: Verify CSV metadata structure and content."""
    frames_dir = temp_dir / "frames"
    processed_dir = temp_dir / "processed"
    config = ExtractionConfig(sampling_interval=5)

    success, summary, df, msg = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=config,
    )
    assert success is True
    assert summary.metadata_path.exists()

    # Read the written CSV from disk
    csv_df = pd.read_csv(summary.metadata_path)
    required_cols = [
        "video_id",
        "frame_index",
        "extracted_frame_index",
        "timestamp_seconds",
        "frame_filename",
        "frame_path",
        "original_fps",
        "sampling_interval",
        "width",
        "height",
    ]
    for col in required_cols:
        assert col in csv_df.columns

    assert csv_df["width"].iloc[0] == 320
    assert csv_df["height"].iloc[0] == 240
    assert csv_df["original_fps"].iloc[0] == 30.0


def test_extract_resizing(test_video, temp_dir):
    """Test optional frame resizing during extraction."""
    frames_dir = temp_dir / "frames"
    processed_dir = temp_dir / "processed"
    config = ExtractionConfig(sampling_interval=10, target_resolution=(160, 120), force_reextract=True)

    success, summary, df, msg = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=config,
    )
    assert success is True
    assert df["width"].iloc[0] == 160
    assert df["height"].iloc[0] == 120

    # Verify actual image on disk has resized dimensions
    first_path = Path(df["frame_path"].iloc[0])
    img = cv2.imread(str(first_path))
    assert img.shape == (120, 160, 3)


def test_extract_caching_and_force_reextract(test_video, temp_dir):
    """Test duplicate extraction avoidance and forced re-extraction."""
    frames_dir = temp_dir / "frames"
    processed_dir = temp_dir / "processed"
    config = ExtractionConfig(sampling_interval=5, force_reextract=False)

    # First run: extracts freshly
    success1, summary1, df1, msg1 = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=config,
    )
    assert success1 is True
    assert summary1.is_cached is False

    # Second run without force: returns cached
    success2, summary2, df2, msg2 = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=config,
    )
    assert success2 is True
    assert summary2.is_cached is True
    assert "existing valid" in msg2.lower()

    # Third run with force_reextract: re-extracts
    config_forced = ExtractionConfig(sampling_interval=5, force_reextract=True)
    success3, summary3, df3, msg3 = extract_video_frames(
        video_path=test_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=config_forced,
    )
    assert success3 is True
    assert summary3.is_cached is False


def test_extract_invalid_inputs(temp_dir):
    """Test 5: Error handling on invalid inputs."""
    frames_dir = temp_dir / "frames"
    processed_dir = temp_dir / "processed"

    # Nonexistent video
    missing = temp_dir / "ghost.mp4"
    success, summary, df, msg = extract_video_frames(
        video_path=missing,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
    )
    assert success is False
    assert "not found" in msg.lower()

    # Invalid sampling interval (<= 0)
    fake_video = temp_dir / "fake.mp4"
    fake_video.touch()
    bad_cfg = ExtractionConfig(sampling_interval=0)
    success, summary, df, msg = extract_video_frames(
        video_path=fake_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
        config=bad_cfg,
    )
    assert success is False
    assert "invalid sampling interval" in msg.lower()

    # Corrupted video file
    corrupt_video = temp_dir / "corrupt.mp4"
    corrupt_video.write_bytes(b"CORRUPTED_HEADER_DATA" * 50)
    success, summary, df, msg = extract_video_frames(
        video_path=corrupt_video,
        output_base_dir=frames_dir,
        processed_base_dir=processed_dir,
    )
    assert success is False
    assert "could not open" in msg.lower() or "unsupported" in msg.lower()
