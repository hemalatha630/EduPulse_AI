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
    """Verify application handles valid video upload end-to-end."""
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
