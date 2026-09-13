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

# Feature 3 Configuration: Student / Person Detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_YOLO_MODEL = "yolov8n.pt"
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"

# Feature 4 Configuration: Student / Person Tracking
DEFAULT_TRACKER = "bytetrack"
SUPPORTED_TRACKERS = ["bytetrack", "botsort"]
DEFAULT_TRACKING_CONF_THRESHOLD = 0.25
DEFAULT_TRAJECTORY_MAX_POINTS = 30
TRACKS_CSV_FILENAME = "tracks.csv"

# Feature 5 Configuration: Observable Behaviour Recognition
DEFAULT_BEHAVIOUR_CONF_THRESHOLD = 0.40
BEHAVIOURS_CSV_FILENAME = "behaviours.csv"
DEFAULT_CROP_SIZE = (224, 224)
MIN_CROP_WIDTH = 20
MIN_CROP_HEIGHT = 30
UNKNOWN_BEHAVIOUR = "Unknown / Uncertain"

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

# High-contrast color palette for behaviour visualization (RGB tuples & Hex strings)
BEHAVIOUR_COLORS = {
    "Looking toward the instructional activity": {"rgb": (46, 204, 113), "hex": "#2ECC71"},  # Green
    "Reading/writing": {"rgb": (52, 152, 219), "hex": "#3498DB"},                            # Blue
    "Interacting with peers": {"rgb": (155, 89, 182), "hex": "#9B59B6"},                     # Purple
    "Looking away": {"rgb": (243, 156, 18), "hex": "#F39C12"},                               # Amber
    "Mobile-device activity": {"rgb": (231, 76, 60), "hex": "#E74C3C"},                       # Red
    "Head-down behaviour": {"rgb": (230, 126, 34), "hex": "#E67E22"},                         # Orange
    "Unknown / Uncertain": {"rgb": (149, 165, 166), "hex": "#95A5A6"},                        # Gray
}

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
