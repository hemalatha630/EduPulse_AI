"""Integration tests for Streamlit app UI using Streamlit AppTest framework."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_FILE = str(ROOT_DIR / "app.py")
SAMPLE_VIDEO = ROOT_DIR / "scratch" / "sample_media" / "sample_classroom.mp4"


@pytest.fixture(scope="module")
def video_bytes():
    """Read sample classroom video bytes."""
    assert SAMPLE_VIDEO.exists(), "Sample video must be generated before running app tests."
    return SAMPLE_VIDEO.read_bytes()


def test_app_initial_render():
    """Verify application renders initial state correctly without errors."""
    at = AppTest.from_file(APP_FILE).run()
    assert not at.exception

    # Check title
    titles = [t.value for t in at.title]
    assert "Temporal Learning-Engagement Profiling from Classroom Videos" in titles

    # Check sidebar research scope
    subheaders = [s.value for s in at.subheader]
    assert "📋 Research Scope" in subheaders

    # Check initial prompt
    infos = [i.value for i in at.info]
    assert any("Please select a classroom video" in info for info in infos)


def test_app_valid_video_upload(video_bytes):
    """Verify application handles valid video upload and renders Feature 2 controls."""
    at = AppTest.from_file(APP_FILE).run()
    at.file_uploader[0].upload(filename="sample_classroom.mp4", content=video_bytes).run()

    assert not at.exception

    # Verify success banner
    successes = [s.value for s in at.success]
    assert any("Video uploaded successfully" in s for s in successes)

    # Verify metrics
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Duration"] == "00:03"
    assert metrics["Resolution"] == "1280 x 720"
    assert "30" in metrics["Frame Rate"]
    assert metrics["Total Frames"] == "90"

    # Verify Feature 2 extraction section rendered
    subheaders = [s.value for s in at.subheader]
    assert any("Feature 2: Frame Extraction & Preprocessing" in s for s in subheaders)

    # Verify extract button is present
    buttons = [b.label for b in at.button]
    assert any("Extract Frames" in b for b in buttons)


def test_app_frame_extraction_workflow(video_bytes):
    """Verify frame extraction button execution in Streamlit AppTest."""
    at = AppTest.from_file(APP_FILE).run()
    at.file_uploader[0].upload(filename="sample_classroom.mp4", content=video_bytes).run()
    assert not at.exception

    # Find and click the extract frames button
    extract_button = next(b for b in at.button if "Extract Frames" in b.label)
    extract_button.click().run()

    assert not at.exception

    # Verify extraction summary subheader
    subheaders = [s.value for s in at.subheader]
    assert any("Frame Extraction Summary" in s for s in subheaders)
    assert any("Preview Extracted Frames" in s for s in subheaders)
    assert any("Temporal Frame Metadata" in s for s in subheaders)

    # Verify metrics rendered in extraction summary
    metrics = {m.label: m.value for m in at.metric}
    assert "Extracted Frames" in metrics
    assert int(metrics["Extracted Frames"].replace(",", "")) > 0


def test_app_detection_workflow(video_bytes):
    """Verify Feature 3 person detection UI workflow end-to-end."""
    at = AppTest.from_file(APP_FILE, default_timeout=30).run()
    at.file_uploader[0].upload(filename="sample_classroom.mp4", content=video_bytes).run()
    assert not at.exception

    # Extract frames first so detection section has frames
    extract_button = next(b for b in at.button if "Extract Frames" in b.label)
    extract_button.click().run()
    assert not at.exception

    # Verify Feature 3 subheader
    subheaders = [s.value for s in at.subheader]
    assert any("Feature 3: Student / Person Detection" in s for s in subheaders)

    # Trigger detection
    detect_btn = next(b for b in at.button if "Detect People in Frames" in b.label)
    detect_btn.click().run()
    assert not at.exception

    # Verify detection summary rendered
    subheaders_after = [s.value for s in at.subheader]
    assert any("Detection Summary" in s for s in subheaders_after)
    assert any("Detection Visualizer" in s for s in subheaders_after)
    assert any("Detections Dataset" in s for s in subheaders_after)


def test_app_tracking_workflow(video_bytes):
    """Verify Feature 4 person tracking UI workflow end-to-end."""
    at = AppTest.from_file(APP_FILE, default_timeout=30).run()
    at.file_uploader[0].upload(filename="sample_classroom.mp4", content=video_bytes).run()
    assert not at.exception

    # Extract frames first so tracking section has frames
    extract_button = next(b for b in at.button if "Extract Frames" in b.label)
    extract_button.click().run()
    assert not at.exception

    # Verify Feature 4 subheader
    subheaders = [s.value for s in at.subheader]
    assert any("Feature 4: Student / Person Tracking" in s for s in subheaders)

    # Trigger tracking
    track_btn = next(b for b in at.button if "Track People Across Consecutive Frames" in b.label)
    track_btn.click().run()
    assert not at.exception

    # Verify tracking summary rendered
    subheaders_after = [s.value for s in at.subheader]
    assert any("Tracking Summary" in s for s in subheaders_after)
    assert any("Sequential Tracking Visualizer" in s for s in subheaders_after)
    assert any("Tracks Dataset" in s for s in subheaders_after)


def test_app_empty_file_upload():
    """Verify application handles empty file gracefully."""
    at = AppTest.from_file(APP_FILE).run()
    at.file_uploader[0].upload(filename="empty_file.mp4", content=b"").run()

    errors = [e.value for e in at.error]
    assert any("empty" in err.lower() for err in errors)


def test_app_corrupted_file_upload():
    """Verify application handles corrupted video file gracefully without crashing."""
    at = AppTest.from_file(APP_FILE).run()
    at.file_uploader[0].upload(
        filename="corrupted_file.mp4", content=b"BAD_CORRUPT_HEADER" * 100
    ).run()

    errors = [e.value for e in at.error]
    assert any("corrupted" in err.lower() or "decode" in err.lower() or "could not open" in err.lower() for err in errors)
