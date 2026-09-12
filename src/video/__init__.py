"""Video processing and frame extraction package for EduPulse AI."""

from src.video.frame_extractor import (
    ExtractionConfig,
    ExtractionSummary,
    FrameMetadata,
    calculate_sampling_info,
    extract_video_frames,
)
from src.video.video_utils import (
    VideoMetadata,
    extract_video_metadata,
    format_duration,
    format_file_size,
    sanitize_filename,
    save_uploaded_video,
    validate_file_extension,
)

__all__ = [
    "VideoMetadata",
    "extract_video_metadata",
    "format_duration",
    "format_file_size",
    "sanitize_filename",
    "save_uploaded_video",
    "validate_file_extension",
    "ExtractionConfig",
    "ExtractionSummary",
    "FrameMetadata",
    "calculate_sampling_info",
    "extract_video_frames",
]
