"""Video processing utilities for validation, metadata extraction, and safe persistence."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Tuple

import cv2

from src.config import SUPPORTED_EXTENSIONS, VIDEOS_DIR, ensure_directories


@dataclass
class VideoMetadata:
    """Structured representation of video technical metadata."""

    filename: str
    file_format: str
    file_size_bytes: int
    file_size_formatted: str
    width: int
    height: int
    resolution: str
    fps: float
    total_frames: int
    duration_seconds: float
    duration_formatted: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a standard dictionary."""
        return asdict(self)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and filesystem conflicts.

    Strips directory navigation sequences, converts backslashes/forward slashes,
    and removes Windows/Unix reserved illegal characters.
    """
    if not filename:
        return "uploaded_video.mp4"

    # Extract only the base name (prevents directory traversal e.g. ../../)
    base_name = Path(filename).name

    # Remove non-alphanumeric, dots, underscores, dashes
    clean_name = re.sub(r'[^\w\.\-]', '_', base_name)

    # Avoid hidden files or leading periods/dashes
    clean_name = clean_name.lstrip('.-')

    # If completely stripped, assign a safe fallback
    if not clean_name:
        clean_name = "classroom_video.mp4"

    return clean_name


def format_file_size(size_in_bytes: int) -> str:
    """Format file size in bytes to a human-readable string (KB, MB, GB)."""
    if size_in_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_in_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_in_bytes / p, 2)
    return f"{s} {units[i]}"


def format_duration(duration_seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS format."""
    if duration_seconds < 0 or math.isnan(duration_seconds) or math.isinf(duration_seconds):
        return "Unavailable"

    total_seconds = int(round(duration_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """Check if the filename has an allowed video extension."""
    suffix = Path(filename).suffix.lower()
    if not suffix:
        return False, "File has no extension. Supported formats: MP4, AVI, MOV, MKV."
    if suffix not in SUPPORTED_EXTENSIONS:
        return (
            False,
            f"Unsupported file type '{suffix}'. Please upload an MP4, AVI, MOV, or MKV video.",
        )
    return True, ""


def extract_video_metadata(video_path: Path) -> Tuple[bool, VideoMetadata | None, str]:
    """Extract technical video metadata using OpenCV.

    Returns:
        tuple (success: bool, metadata: VideoMetadata or None, error_message: str)
    """
    if not video_path.exists():
        return False, None, f"Video file not found at: {video_path.name}"

    file_size = video_path.stat().st_size
    if file_size == 0:
        return False, None, "The uploaded file is empty (0 bytes)."

    # Check extension
    is_valid_ext, ext_err = validate_file_extension(video_path.name)
    if not is_valid_ext:
        return False, None, ext_err

    # Attempt to open video with OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path.resolve()))
    except Exception as exc:
        return False, None, f"Failed to initialize video capture: {str(exc)}"

    if not cap.isOpened():
        cap.release()
        return (
            False,
            None,
            "Could not open the video file. The file may be corrupted, truncated, or encoded with an unsupported codec.",
        )

    # Verify that at least the first frame can be successfully read and decoded
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        cap.release()
        return (
            False,
            None,
            "Failed to decode video frames. The file is unreadable or corrupted.",
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle unreliable metadata cases
        if fps <= 0 or math.isnan(fps) or math.isinf(fps):
            fps_val = 0.0
            duration_sec = 0.0
            duration_str = "Unavailable"
        else:
            fps_val = round(fps, 2)
            if total_frames > 0:
                duration_sec = total_frames / fps_val
                duration_str = format_duration(duration_sec)
            else:
                duration_sec = 0.0
                duration_str = "Unavailable"

        resolution_str = f"{width} x {height}" if width > 0 and height > 0 else "Unavailable"
        file_format = video_path.suffix.lstrip('.').upper()

        metadata = VideoMetadata(
            filename=video_path.name,
            file_format=file_format,
            file_size_bytes=file_size,
            file_size_formatted=format_file_size(file_size),
            width=width,
            height=height,
            resolution=resolution_str,
            fps=fps_val,
            total_frames=total_frames if total_frames >= 0 else 0,
            duration_seconds=round(duration_sec, 2),
            duration_formatted=duration_str,
        )
        return True, metadata, ""
    except Exception as exc:
        return False, None, f"Error calculating video properties: {str(exc)}"
    finally:
        cap.release()


def save_uploaded_video(
    uploaded_file, target_dir: Path | None = None
) -> Tuple[bool, Path | None, VideoMetadata | None, str]:
    """Save an uploaded Streamlit video buffer to disk safely, validate it, and extract metadata.

    Args:
        uploaded_file: Streamlit UploadedFile object or file-like buffer
        target_dir: Destination folder (defaults to VIDEOS_DIR)

    Returns:
        tuple: (success: bool, saved_path: Path or None, metadata: VideoMetadata or None, message: str)
    """
    if uploaded_file is None:
        return False, None, None, "No file was uploaded."

    destination_dir = target_dir or VIDEOS_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    ensure_directories()

    # Step 1: Validate Extension
    raw_name = getattr(uploaded_file, "name", "uploaded_video.mp4")
    is_valid_ext, ext_err = validate_file_extension(raw_name)
    if not is_valid_ext:
        return False, None, None, ext_err

    # Step 2: Sanitize file name
    safe_name = sanitize_filename(raw_name)
    destination_path = destination_dir / safe_name

    # Step 3: Stream content to destination file
    try:
        # Check size if available
        file_size = getattr(uploaded_file, "size", None)
        if file_size == 0:
            return False, None, None, "The uploaded file is empty (0 bytes)."

        # Write chunks to disk
        bytes_written = 0
        with open(destination_path, "wb") as f:
            if hasattr(uploaded_file, "getbuffer"):
                buffer = uploaded_file.getbuffer()
                if len(buffer) == 0:
                    destination_path.unlink(missing_ok=True)
                    return False, None, None, "The uploaded file is empty (0 bytes)."
                f.write(buffer)
                bytes_written = len(buffer)
            else:
                data = uploaded_file.read()
                if not data:
                    destination_path.unlink(missing_ok=True)
                    return False, None, None, "The uploaded file is empty (0 bytes)."
                f.write(data)
                bytes_written = len(data)

        if bytes_written == 0:
            destination_path.unlink(missing_ok=True)
            return False, None, None, "The uploaded file is empty (0 bytes)."

    except Exception as exc:
        destination_path.unlink(missing_ok=True)
        return False, None, None, f"Failed to save video to storage: {str(exc)}"

    # Step 4: Validate video content using OpenCV and extract metadata
    success, metadata, err_msg = extract_video_metadata(destination_path)
    if not success:
        # Clean up invalid/corrupted file to protect workspace storage
        destination_path.unlink(missing_ok=True)
        return False, None, None, err_msg

    return True, destination_path, metadata, "Video uploaded successfully."
