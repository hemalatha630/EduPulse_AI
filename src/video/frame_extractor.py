"""Frame extraction module for classroom video analysis.

Extracts chronological frames with configurable temporal sampling,
generates structured temporal metadata, and saves frames and CSV records.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FRAME_FILENAME_PATTERN,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_SAMPLING_INTERVAL,
    FRAMES_DIR,
    PROCESSED_DIR,
)
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.video.video_utils import format_duration, sanitize_filename


@dataclass
class ExtractionConfig:
    """Configuration settings for frame extraction and preprocessing."""

    sampling_interval: int = DEFAULT_SAMPLING_INTERVAL
    target_resolution: Tuple[int, int] | None = None
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    force_reextract: bool = False


@dataclass
class FrameMetadata:
    """Metadata record for an individual extracted video frame."""

    video_id: str
    frame_index: int
    extracted_frame_index: int
    timestamp_seconds: float
    frame_filename: str
    frame_path: str
    original_fps: float
    sampling_interval: int
    width: int
    height: int

    def to_dict(self) -> dict:
        """Convert dataclass to standard dictionary."""
        return asdict(self)


@dataclass
class ExtractionSummary:
    """Summary of the frame extraction execution."""

    video_id: str
    original_fps: float
    original_total_frames: int
    duration_seconds: float
    duration_formatted: str
    sampling_description: str
    sampling_interval: int
    effective_fps: float
    extracted_frames_count: int
    frames_directory: Path
    metadata_path: Path
    status: str
    is_cached: bool = False

    def to_dict(self) -> dict:
        """Convert dataclass to standard dictionary."""
        data = asdict(self)
        data["frames_directory"] = str(self.frames_directory)
        data["metadata_path"] = str(self.metadata_path)
        return data


def derive_video_id(video_path: Path) -> str:
    """Generate a clean, safe video identifier based on the file stem."""
    raw_stem = video_path.stem
    safe_stem = sanitize_filename(f"{raw_stem}.mp4")
    video_id = Path(safe_stem).stem
    return video_id or "classroom_video"


def calculate_sampling_info(original_fps: float, sampling_interval: int) -> Tuple[float, str]:
    """Calculate effective sampling rate (FPS) and descriptive text.

    Args:
        original_fps: Original video frames per second.
        sampling_interval: Frame stride (1 = every frame, 5 = every 5th frame).

    Returns:
        Tuple of (effective_fps: float, sampling_description: str)
    """
    if sampling_interval <= 0:
        raise ValueError(f"Sampling interval must be positive, got {sampling_interval}")

    if sampling_interval == 1:
        desc = "Every frame (1:1)"
        effective_fps = original_fps
    else:
        desc = f"Every {sampling_interval}th frame"
        effective_fps = (original_fps / sampling_interval) if original_fps > 0 else 0.0

    return round(effective_fps, 2), desc


def check_existing_extraction(
    frames_dir: Path,
    metadata_path: Path,
    expected_interval: int | None = None,
    expected_resolution: Tuple[int, int] | None = None,
) -> Tuple[bool, pd.DataFrame | None]:
    """Verify if a valid prior frame extraction exists on disk.

    Args:
        frames_dir: Directory containing extracted image frames.
        metadata_path: Path to the frame metadata CSV.
        expected_interval: Optional expected sampling interval.
        expected_resolution: Optional expected (width, height) resolution.

    Returns:
        Tuple of (exists_and_valid: bool, metadata_df: pd.DataFrame or None)
    """
    if not frames_dir.exists() or not metadata_path.exists():
        return False, None

    try:
        df = pd.read_csv(metadata_path)
        if df.empty:
            return False, None

        required_columns = {
            "video_id",
            "frame_index",
            "extracted_frame_index",
            "timestamp_seconds",
            "frame_path",
            "original_fps",
            "sampling_interval",
        }
        if not required_columns.issubset(df.columns):
            return False, None

        # Check sampling interval match if specified
        if expected_interval is not None:
            if "sampling_interval" in df.columns:
                if (df["sampling_interval"] != expected_interval).any():
                    return False, None

        # Check resolution match if specified
        if expected_resolution is not None and "width" in df.columns and "height" in df.columns:
            exp_w, exp_h = expected_resolution
            if (df["width"] != exp_w).any() or (df["height"] != exp_h).any():
                return False, None

        # Check that at least the first and last frame files physically exist
        first_frame = Path(df.iloc[0]["frame_path"])
        last_frame = Path(df.iloc[-1]["frame_path"])
        if not first_frame.exists() or not last_frame.exists():
            return False, None

        return True, df
    except Exception:
        return False, None


def extract_video_frames(
    video_path: Path,
    output_base_dir: Path | None = None,
    processed_base_dir: Path | None = None,
    config: ExtractionConfig | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Tuple[bool, ExtractionSummary | None, pd.DataFrame | None, str]:
    """Extract frames chronologically from video with sampling and temporal metadata.

    Args:
        video_path: Path to the source video file.
        output_base_dir: Base directory for frames (defaults to FRAMES_DIR).
        processed_base_dir: Base directory for metadata (defaults to PROCESSED_DIR).
        config: Extraction configuration settings.
        progress_callback: Optional callback receiving (current_frame, total_frames, status_msg).

    Returns:
        Tuple of (success: bool, summary: ExtractionSummary or None, df: pd.DataFrame or None, message: str)
    """
    cfg = config or ExtractionConfig()

    if not video_path.exists():
        return False, None, None, f"Video file not found at: {video_path.name}"

    if cfg.sampling_interval <= 0:
        return (
            False,
            None,
            None,
            f"Invalid sampling interval: {cfg.sampling_interval}. Must be a positive integer >= 1.",
        )

    video_id = derive_video_id(video_path)
    frames_dir = (output_base_dir or FRAMES_DIR) / video_id
    processed_dir = (processed_base_dir or PROCESSED_DIR) / video_id
    metadata_csv_path = processed_dir / "frame_metadata.csv"

    # Step 1: Check if valid extracted frames already exist (unless force_reextract is requested)
    if not cfg.force_reextract:
        is_cached, cached_df = check_existing_extraction(
            frames_dir=frames_dir,
            metadata_path=metadata_csv_path,
            expected_interval=cfg.sampling_interval,
            expected_resolution=cfg.target_resolution,
        )
        if is_cached and cached_df is not None:
            first_row = cached_df.iloc[0]
            last_row = cached_df.iloc[-1]
            orig_fps = float(first_row.get("original_fps", 30.0))
            eff_fps, samp_desc = calculate_sampling_info(orig_fps, cfg.sampling_interval)
            duration_sec = float(last_row.get("timestamp_seconds", 0.0))

            summary = ExtractionSummary(
                video_id=video_id,
                original_fps=orig_fps,
                original_total_frames=int(cached_df["frame_index"].max() + 1),
                duration_seconds=round(duration_sec, 2),
                duration_formatted=format_duration(duration_sec),
                sampling_description=samp_desc,
                sampling_interval=cfg.sampling_interval,
                effective_fps=eff_fps,
                extracted_frames_count=len(cached_df),
                frames_directory=frames_dir,
                metadata_path=metadata_csv_path,
                status="Completed (Cached)",
                is_cached=True,
            )
            return True, summary, cached_df, "Using existing valid extracted frames."

    # Step 2: Open video using OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path.resolve()))
    except Exception as exc:
        return False, None, None, f"Failed to initialize video capture: {str(exc)}"

    if not cap.isOpened():
        cap.release()
        return (
            False,
            None,
            None,
            "Could not open the video file for extraction. File may be corrupted or codec unsupported.",
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or math.isnan(fps) or math.isinf(fps):
            fps = 30.0  # Fallback default if stream does not report FPS

        effective_fps, sampling_desc = calculate_sampling_info(fps, cfg.sampling_interval)

        # Prepare directories
        frames_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Remove old frames if re-extracting
        if cfg.force_reextract and frames_dir.exists():
            for old_frame in frames_dir.glob("*.jpg"):
                try:
                    old_frame.unlink(missing_ok=True)
                except OSError:
                    pass

        metadata_records: list[dict] = []
        raw_frame_idx = 0
        extracted_frame_idx = 0

        total_to_process = max(total_frames, 1)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Apply sampling filter
            if raw_frame_idx % cfg.sampling_interval == 0:
                is_valid, val_err = FramePreprocessor.validate_frame(frame)
                if not is_valid:
                    cap.release()
                    return False, None, None, f"Frame {raw_frame_idx} is invalid: {val_err}"

                # Apply optional resizing
                processed_frame = FramePreprocessor.resize_frame(frame, cfg.target_resolution)

                extracted_frame_idx += 1
                frame_filename = DEFAULT_FRAME_FILENAME_PATTERN.format(index=extracted_frame_idx)
                frame_output_path = frames_dir / frame_filename

                save_success, save_err = FramePreprocessor.save_frame(
                    processed_frame, frame_output_path, quality=cfg.jpeg_quality
                )
                if not save_success:
                    cap.release()
                    return False, None, None, f"Storage error writing frame {extracted_frame_idx}: {save_err}"

                # Compute temporal timestamp
                timestamp_seconds = round(raw_frame_idx / fps, 3) if fps > 0 else 0.0
                frame_h, frame_w = processed_frame.shape[:2]

                # Relative path for cross-platform portability
                rel_frame_path = str(frame_output_path.relative_to(output_base_dir or FRAMES_DIR.parent))

                record = FrameMetadata(
                    video_id=video_id,
                    frame_index=raw_frame_idx,
                    extracted_frame_index=extracted_frame_idx,
                    timestamp_seconds=timestamp_seconds,
                    frame_filename=frame_filename,
                    frame_path=str(frame_output_path.resolve()),
                    original_fps=round(fps, 2),
                    sampling_interval=cfg.sampling_interval,
                    width=frame_w,
                    height=frame_h,
                )
                metadata_records.append(record.to_dict())

            raw_frame_idx += 1

            # Update progress callback periodically
            if progress_callback is not None and (raw_frame_idx % 5 == 0 or raw_frame_idx == total_to_process):
                progress_callback(
                    raw_frame_idx,
                    total_to_process,
                    f"Processing video frames ({raw_frame_idx}/{total_to_process})...",
                )

    except Exception as exc:
        return False, None, None, f"Unexpected error during frame extraction: {str(exc)}"
    finally:
        cap.release()

    if extracted_frame_idx == 0:
        return (
            False,
            None,
            None,
            "No frames could be extracted. Please check video content and sampling settings.",
        )

    # Step 3: Write metadata DataFrame to CSV
    try:
        df = pd.DataFrame(metadata_records)
        df.to_csv(metadata_csv_path, index=False)
    except Exception as exc:
        return False, None, None, f"Failed to save frame metadata CSV: {str(exc)}"

    reported_total_frames = total_frames if total_frames > 0 else raw_frame_idx
    duration_sec = round(reported_total_frames / fps, 2) if fps > 0 else 0.0

    summary = ExtractionSummary(
        video_id=video_id,
        original_fps=round(fps, 2),
        original_total_frames=reported_total_frames,
        duration_seconds=duration_sec,
        duration_formatted=format_duration(duration_sec),
        sampling_description=sampling_desc,
        sampling_interval=cfg.sampling_interval,
        effective_fps=effective_fps,
        extracted_frames_count=extracted_frame_idx,
        frames_directory=frames_dir,
        metadata_path=metadata_csv_path,
        status="Completed",
        is_cached=False,
    )

    if progress_callback:
        progress_callback(total_to_process, total_to_process, "Frame extraction complete!")

    return True, summary, df, "Frame extraction completed successfully."
