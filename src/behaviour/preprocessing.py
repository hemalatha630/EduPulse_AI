"""Person Crop Preprocessing Module for Behaviour Recognition.

Provides robust, safe extraction, boundary clipping, resizing, and normalization
of tracked person crops for observable behaviour classification.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch

from src.config import (
    DEFAULT_CROP_SIZE,
    MIN_CROP_HEIGHT,
    MIN_CROP_WIDTH,
)

# Standard ImageNet normalization parameters for PyTorch visual models
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_person_crop(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    target_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
) -> Tuple[bool, Optional[np.ndarray], Optional[torch.Tensor], str]:
    """Safely extract, clip, resize, and normalize a tracked person region.

    Args:
        frame: Full video frame as a NumPy array (RGB, uint8).
        bbox: Bounding box tuple (x1, y1, x2, y2).
        target_size: Target (width, height) for resized crop (default: 224x224).

    Returns:
        Tuple containing:
        - success (bool): True if valid crop extracted, False otherwise.
        - crop_rgb (Optional[np.ndarray]): Resized RGB image of shape (H, W, 3).
        - crop_tensor (Optional[torch.Tensor]): Normalized PyTorch tensor of shape (1, 3, H, W).
        - error_message (str): Explanatory message if extraction failed, or empty string.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False, None, None, "Invalid or empty input frame."

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return False, None, None, f"Frame must have shape (H, W, 3), got {frame.shape}."

    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Check for NaN or Inf
    if not all(np.isfinite([x1, y1, x2, y2])):
        return False, None, None, f"Non-finite bounding box coordinates: {bbox}."

    # Clip coordinates safely to frame boundaries
    x1_clipped = max(0, min(int(round(x1)), w_frame - 1))
    y1_clipped = max(0, min(int(round(y1)), h_frame - 1))
    x2_clipped = max(0, min(int(round(x2)), w_frame))
    y2_clipped = max(0, min(int(round(y2)), h_frame))

    # Verify positive dimensions
    crop_w = x2_clipped - x1_clipped
    crop_h = y2_clipped - y1_clipped

    if crop_w <= 0 or crop_h <= 0:
        return False, None, None, (
            f"Invalid bounding box geometry after boundary clipping: "
            f"width={crop_w}, height={crop_h} from original bbox={bbox}."
        )

    # Check minimum resolution thresholds for reliable visual assessment
    if crop_w < MIN_CROP_WIDTH or crop_h < MIN_CROP_HEIGHT:
        return False, None, None, (
            f"Person crop is too small for classification: {crop_w}x{crop_h} px "
            f"(minimum required: {MIN_CROP_WIDTH}x{MIN_CROP_HEIGHT} px)."
        )

    # Extract crop
    crop = frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
    if crop.size == 0:
        return False, None, None, "Extracted crop slice is empty."

    # Resize to standard model input dimensions
    target_w, target_h = target_size
    crop_resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert to normalized PyTorch tensor: (H, W, C) -> (C, H, W), float32 in [0, 1]
    norm_img = crop_resized.astype(np.float32) / 255.0
    norm_img = (norm_img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float()

    return True, crop_resized, tensor, ""
