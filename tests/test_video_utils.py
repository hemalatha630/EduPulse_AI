"""Unit tests for video validation, sanitization, and metadata extraction."""

from io import BytesIO
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from src.video.video_utils import (
    extract_video_metadata,
    format_duration,
    format_file_size,
    sanitize_filename,
    save_uploaded_video,
    validate_file_extension,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def synthetic_video(temp_dir) -> Path:
    """Create a small valid MP4 video using OpenCV VideoWriter."""
    video_path = temp_dir / "sample_test_video.mp4"
    width, height = 320, 240
    fps = 20.0
    num_frames = 20  # exactly 1.0 second

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a simple synthetic color frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a moving color square
        color = ((i * 12) % 255, (i * 20) % 255, (i * 30) % 255)
        cv2.rectangle(frame, (20 + i * 5, 20), (100 + i * 5, 100), color, -1)
        out.write(frame)

    out.release()
    return video_path


class MockUploadedFile:
    """Mock Streamlit UploadedFile interface for unit testing."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)

    def getbuffer(self):
        return memoryview(self._data)

    def read(self):
        return self._data


# =====================================================================
# Tests: Filename Sanitization & Extension Validation
# =====================================================================


def test_sanitize_filename_normal():
    assert sanitize_filename("classroom_01.mp4") == "classroom_01.mp4"


def test_sanitize_filename_traversal():
    assert sanitize_filename("../../etc/passwd.mp4") == "passwd.mp4"
    assert sanitize_filename("..\\..\\windows\\system32\\secret.avi") == "secret.avi"


def test_sanitize_filename_special_chars():
    result = sanitize_filename("Lecture #1 [Section A] & (Group B).mov")
    assert ".." not in result
    assert "/" not in result
    assert "\\" not in result
    assert result.endswith(".mov")


def test_sanitize_filename_empty():
    assert sanitize_filename("") == "uploaded_video.mp4"
    assert sanitize_filename("...") == "classroom_video.mp4"


def test_validate_file_extension():
    # Valid
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]:
        is_valid, _ = validate_file_extension(f"test_video{ext}")
        assert is_valid is True

    # Invalid
    for name in ["document.pdf", "script.py", "audio.mp3", "image.png", "no_ext"]:
        is_valid, err = validate_file_extension(name)
        assert is_valid is False
        assert len(err) > 0


# =====================================================================
# Tests: Duration and File Size Formatting
# =====================================================================


def test_format_file_size():
    assert format_file_size(0) == "0 B"
    assert format_file_size(500) == "500.0 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1024 * 1024 * 5) == "5.0 MB"


def test_format_duration():
    assert format_duration(45.0) == "00:45"
    assert format_duration(155.0) == "02:35"
    assert format_duration(3665.0) == "01:01:05"
    assert format_duration(-1.0) == "Unavailable"


# =====================================================================
# Tests: Metadata Extraction & OpenCV Validation
# =====================================================================


def test_extract_video_metadata_valid(synthetic_video):
    success, metadata, err = extract_video_metadata(synthetic_video)
    assert success is True
    assert err == ""
    assert metadata is not None
    assert metadata.filename == "sample_test_video.mp4"
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.resolution == "320 x 240"
    assert metadata.fps == 20.0
    assert metadata.total_frames == 20
    assert metadata.duration_seconds == 1.0
    assert metadata.duration_formatted == "00:01"
    assert metadata.file_format == "MP4"


def test_extract_video_metadata_empty_file(temp_dir):
    empty_file = temp_dir / "empty.mp4"
    empty_file.touch()

    success, metadata, err = extract_video_metadata(empty_file)
    assert success is False
    assert metadata is None
    assert "empty" in err.lower()


def test_extract_video_metadata_corrupted_file(temp_dir):
    corrupt_file = temp_dir / "corrupted.mp4"
    # Write random invalid binary bytes
    corrupt_file.write_bytes(b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99" * 50)

    success, metadata, err = extract_video_metadata(corrupt_file)
    assert success is False
    assert metadata is None
    assert "corrupt" in err.lower() or "decode" in err.lower() or "could not open" in err.lower()


def test_extract_video_metadata_nonexistent(temp_dir):
    missing_file = temp_dir / "ghost.mp4"
    success, metadata, err = extract_video_metadata(missing_file)
    assert success is False
    assert metadata is None
    assert "not found" in err.lower()


# =====================================================================
# Tests: Save Uploaded Video Workflow
# =====================================================================


def test_save_uploaded_video_valid(synthetic_video, temp_dir):
    video_bytes = synthetic_video.read_bytes()
    mock_upload = MockUploadedFile("my_lecture.mp4", video_bytes)

    target_dir = temp_dir / "videos"
    success, saved_path, metadata, msg = save_uploaded_video(mock_upload, target_dir)

    assert success is True
    assert saved_path is not None
    assert saved_path.exists()
    assert saved_path.name == "my_lecture.mp4"
    assert metadata is not None
    assert metadata.width == 320
    assert "successfully" in msg.lower()


def test_save_uploaded_video_empty(temp_dir):
    mock_upload = MockUploadedFile("empty_lecture.mp4", b"")
    target_dir = temp_dir / "videos"

    success, saved_path, metadata, msg = save_uploaded_video(mock_upload, target_dir)

    assert success is False
    assert saved_path is None
    assert metadata is None
    assert "empty" in msg.lower()


def test_save_uploaded_video_corrupt(temp_dir):
    mock_upload = MockUploadedFile("corrupt.mp4", b"BAD_VIDEO_HEADER_DATA_NOT_A_REAL_VIDEO")
    target_dir = temp_dir / "videos"

    success, saved_path, metadata, msg = save_uploaded_video(mock_upload, target_dir)

    assert success is False
    assert saved_path is None
    assert metadata is None
    # Verify corrupted file was cleaned up from storage
    assert not (target_dir / "corrupt.mp4").exists()
