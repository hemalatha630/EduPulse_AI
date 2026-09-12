"""Configuration and path definitions for EduPulse AI.

Provides centralized path management, supported format definitions,
and project metadata constants.
"""

from pathlib import Path

# Base Paths (dynamic, relative to workspace root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Video Configuration
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB

# Feature 2 Configuration: Frame Extraction & Preprocessing
DEFAULT_SAMPLING_INTERVAL = 5  # Every 5th frame (~6 FPS for 30 FPS video)
DEFAULT_JPEG_QUALITY = 95
DEFAULT_FRAME_FILENAME_PATTERN = "frame_{index:06d}.jpg"

# Project Metadata
PROJECT_TITLE = "Temporal Learning-Engagement Profiling from Classroom Videos"
PROJECT_SHORT_TITLE = "EduPulse AI"

# Research Scope: Observable Learning-Related Behaviours
TARGET_OBSERVABLE_BEHAVIOURS = [
    "Looking toward the instructional activity",
    "Reading/writing",
    "Interacting with peers",
    "Looking away",
    "Mobile-device activity",
    "Head-down behaviour",
]

# Explicitly Excluded Claims (Scientific rigor & ethical boundaries)
EXCLUDED_INTERNAL_STATES = [
    "Emotions (happiness, sadness, etc.)",
    "Boredom / motivation",
    "Intelligence / understanding",
    "Internal mental engagement or cognitive states",
]


def ensure_directories() -> None:
    """Ensure all required project data directories exist on disk."""
    for directory in [VIDEOS_DIR, FRAMES_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
