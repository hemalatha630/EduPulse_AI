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

# Canonical Behaviour Class Constants
CLASS_LOOKING_TOWARD_INSTRUCTION = "Looking toward the instructional activity"
CLASS_READING_WRITING = "Reading/writing"
CLASS_INTERACTING_WITH_PEERS = "Interacting with peers"
CLASS_LOOKING_AWAY = "Looking away"
CLASS_MOBILE_DEVICE_ACTIVITY = "Mobile-device activity"
CLASS_HEAD_DOWN = "Head-down behaviour"
CLASS_UNKNOWN = UNKNOWN_BEHAVIOUR

# Feature 6 Configuration: CNN Visual Feature Extraction
DEFAULT_CNN_MODEL = "resnet18"
SUPPORTED_CNN_MODELS = ["resnet18"]
CNN_FEATURE_DIM = 512
CNN_FEATURES_NPY_FILENAME = "cnn_features.npy"
CNN_METADATA_CSV_FILENAME = "cnn_features_metadata.csv"
DEFAULT_CNN_BATCH_SIZE = 16

# Feature 7 Configuration: Temporal Sequence Creation
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_SEQUENCE_STRIDE = 2
DEFAULT_MAX_FRAME_GAP = 2
TEMPORAL_SEQUENCES_NPY_FILENAME = "temporal_sequences.npy"
TEMPORAL_SEQUENCES_METADATA_FILENAME = "temporal_sequences_metadata.csv"

# Feature 8 Configuration: RNN / LSTM / GRU Temporal Modelling
MODELS_TEMPORAL_DIR = MODELS_DIR / "temporal"
RESULTS_TEMPORAL_DIR = RESULTS_DIR / "temporal"
MODEL_RNN = "rnn"
MODEL_LSTM = "lstm"
MODEL_GRU = "gru"
SUPPORTED_TEMPORAL_MODELS = [MODEL_RNN, MODEL_LSTM, MODEL_GRU]
DEFAULT_TEMPORAL_MODEL = MODEL_LSTM
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TEMPORAL_BATCH_SIZE = 8
DEFAULT_BATCH_SIZE = DEFAULT_TEMPORAL_BATCH_SIZE
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 5
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42
PREDICTIONS_CSV_FILENAME = "predictions.csv"

# Project Metadata
PROJECT_TITLE = "Temporal Learning-Engagement Profiling from Classroom Videos"
PROJECT_SHORT_TITLE = "EduPulse AI"

# Research Scope: Observable Learning-Related Behaviours
TARGET_OBSERVABLE_BEHAVIOURS = [
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_READING_WRITING,
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_AWAY,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_HEAD_DOWN,
]

ALL_OBSERVABLE_BEHAVIOURS = TARGET_OBSERVABLE_BEHAVIOURS + [CLASS_UNKNOWN]

# High-contrast color palette for behaviour visualization (RGB tuples & Hex strings)
BEHAVIOUR_COLORS = {
    CLASS_LOOKING_TOWARD_INSTRUCTION: {"rgb": (46, 204, 113), "hex": "#2ECC71"},  # Green
    CLASS_READING_WRITING: {"rgb": (52, 152, 219), "hex": "#3498DB"},             # Blue
    CLASS_INTERACTING_WITH_PEERS: {"rgb": (155, 89, 182), "hex": "#9B59B6"},      # Purple
    CLASS_LOOKING_AWAY: {"rgb": (243, 156, 18), "hex": "#F39C12"},                # Amber
    CLASS_MOBILE_DEVICE_ACTIVITY: {"rgb": (231, 76, 60), "hex": "#E74C3C"},        # Red
    CLASS_HEAD_DOWN: {"rgb": (230, 126, 34), "hex": "#E67E22"},                  # Orange
    CLASS_UNKNOWN: {"rgb": (149, 165, 166), "hex": "#95A5A6"},                    # Gray
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
    for directory in [
        VIDEOS_DIR,
        FRAMES_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        MODELS_TEMPORAL_DIR,
        RESULTS_TEMPORAL_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
