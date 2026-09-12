"""Frame preprocessing module for classroom video analysis.

Provides frame validation, color space conversions (BGR <-> RGB),
configurable resizing, and safe image persistence.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.config import DEFAULT_JPEG_QUALITY


class FramePreprocessor:
    """Handles low-level frame validation, color conversions, and image transformations."""

    @staticmethod
    def validate_frame(frame: np.ndarray | None) -> Tuple[bool, str]:
        """Validate that an extracted frame is valid, non-empty, and correctly shaped.

        Args:
            frame: Numpy array representing an extracted image frame.

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if frame is None:
            return False, "Frame object is None."

        if not isinstance(frame, np.ndarray):
            return False, f"Expected numpy.ndarray frame, got {type(frame).__name__}."

        if frame.size == 0:
            return False, "Frame data is empty (size 0 bytes)."

        if len(frame.shape) != 3:
            return False, f"Expected 3-dimensional frame (H, W, C), got shape {frame.shape}."

        height, width, channels = frame.shape
        if height <= 0 or width <= 0:
            return False, f"Invalid frame dimensions: width={width}, height={height}."

        if channels not in (1, 3, 4):
            return False, f"Unsupported channel count: {channels}. Expected 1, 3, or 4 channels."

        return True, ""

    @staticmethod
    def resize_frame(
        frame: np.ndarray, target_size: Tuple[int, int] | None = None
    ) -> np.ndarray:
        """Resize a frame to the target (width, height) resolution if specified.

        Uses INTER_AREA when downsampling for anti-aliasing quality,
        and INTER_LINEAR when upscaling.

        Args:
            frame: Input image array (H, W, C).
            target_size: Optional (width, height) tuple. If None, original frame is returned.

        Returns:
            Resized or original frame.
        """
        if target_size is None:
            return frame

        target_w, target_h = target_size
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Target dimensions must be positive integers, got: {target_size}")

        current_h, current_w = frame.shape[:2]
        if current_w == target_w and current_h == target_h:
            return frame

        # Use area interpolation when downscaling, linear when upscaling
        if target_w < current_w or target_h < current_h:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert a BGR image (OpenCV default) to RGB (for Streamlit display).

        Args:
            frame: Input BGR image array.

        Returns:
            RGB image array.
        """
        if frame is None or frame.size == 0:
            return frame
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
        """Convert an RGB image to BGR (for OpenCV file persistence).

        Args:
            frame: Input RGB image array.

        Returns:
            BGR image array.
        """
        if frame is None or frame.size == 0:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def save_frame(
        frame: np.ndarray,
        output_path: Path,
        quality: int = DEFAULT_JPEG_QUALITY,
    ) -> Tuple[bool, str]:
        """Save a BGR frame to disk as a standardized JPEG image.

        Args:
            frame: Frame array in BGR format.
            output_path: Destination file path (must have image extension, e.g. .jpg).
            quality: JPEG quality level (1-100). Default is 95.

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        is_valid, err_msg = FramePreprocessor.validate_frame(frame)
        if not is_valid:
            return False, f"Cannot save invalid frame: {err_msg}"

        if not (1 <= quality <= 100):
            return False, f"JPEG quality must be between 1 and 100, got: {quality}"

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success = cv2.imwrite(str(output_path.resolve()), frame, encode_params)

            if not success or not output_path.exists() or output_path.stat().st_size == 0:
                return False, f"Failed to write image data to {output_path.name}"

            return True, ""
        except Exception as exc:
            return False, f"Filesystem error saving frame {output_path.name}: {str(exc)}"

    @staticmethod
    def load_frame(frame_path: Path) -> Tuple[bool, np.ndarray | None, str]:
        """Load an image frame from disk in BGR format and validate its readability.

        Args:
            frame_path: Path to the image file.

        Returns:
            Tuple of (success: bool, frame: np.ndarray or None, error_message: str)
        """
        if not frame_path.exists():
            return False, None, f"Image file not found: {frame_path.name}"

        try:
            frame = cv2.imread(str(frame_path.resolve()), cv2.IMREAD_COLOR)
            is_valid, err = FramePreprocessor.validate_frame(frame)
            if not is_valid:
                return False, None, f"Corrupted or unreadable image {frame_path.name}: {err}"
            return True, frame, ""
        except Exception as exc:
            return False, None, f"Error reading frame {frame_path.name}: {str(exc)}"
