"""CNN Visual Feature Extraction Module for EduPulse AI.

Extracts fixed-length numerical feature vectors from tracked person crops using
a pretrained CNN backbone (ResNet18) without classification heads.
Preserves frame/track/timestamp temporal metadata and links observable behaviour labels,
saving feature arrays to .npy and corresponding index mappings to metadata .csv.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from src.behaviour.preprocessing import preprocess_person_crop
from src.config import (
    CNN_FEATURE_DIM,
    CNN_FEATURES_NPY_FILENAME,
    CNN_METADATA_CSV_FILENAME,
    DEFAULT_CNN_BATCH_SIZE,
    DEFAULT_CNN_MODEL,
    FRAMES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
)


@dataclass
class ExtractionSummary:
    """Summary metrics of a CNN feature extraction session."""

    video_id: str
    model_name: str
    device: str
    feature_dimension: int
    total_frames_processed: int
    total_crops_evaluated: int
    valid_features_extracted: int
    skipped_crops_count: int
    processing_time_seconds: float
    features_npy_path: Path
    metadata_csv_path: Path
    status: str

    def to_dict(self) -> dict:
        """Convert summary to serializable dictionary."""
        return {
            "video_id": self.video_id,
            "model_name": self.model_name,
            "device": self.device,
            "feature_dimension": self.feature_dimension,
            "total_frames_processed": self.total_frames_processed,
            "total_crops_evaluated": self.total_crops_evaluated,
            "valid_features_extracted": self.valid_features_extracted,
            "skipped_crops_count": self.skipped_crops_count,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "features_npy_path": str(self.features_npy_path),
            "metadata_csv_path": str(self.metadata_csv_path),
            "status": self.status,
        }


class CNNFeatureExtractor:
    """Pretrained CNN visual feature extractor for tracked student crops.

    Replaces the final classification head (fc layer) with nn.Identity() to output
    raw 512-dimensional embeddings capturing visual patterns such as body posture,
    head orientation, visible objects/desks, and spatial appearance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CNN_MODEL,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.feature_dim = CNN_FEATURE_DIM
        self._load_model()

    def _load_model(self) -> None:
        """Initialize and prepare the pretrained CNN backbone."""
        if self.model_name.lower() == "resnet18":
            try:
                # Load torchvision ResNet18 with default ImageNet pretrained weights
                weights = models.ResNet18_Weights.DEFAULT
                base_model = models.resnet18(weights=weights)
            except Exception as exc:
                # Fallback in offline environments or weight download failures
                try:
                    base_model = models.resnet18(weights=None)
                except Exception as inner_exc:
                    raise RuntimeError(
                        f"Failed to initialize ResNet18 architecture: {inner_exc}"
                    ) from exc

            # Strip final classification layer to produce 512-dim visual representation
            base_model.fc = nn.Identity()
            base_model.to(self.device)
            base_model.eval()
            self.model = base_model
            self.feature_dim = CNN_FEATURE_DIM
        else:
            raise ValueError(
                f"Unsupported CNN model '{self.model_name}'. Supported models: ['resnet18']"
            )

    @property
    def device_name(self) -> str:
        """Return standardized hardware device indicator string."""
        return "CUDA" if self.device.type == "cuda" else "CPU"

    def extract_single(self, crop_tensor: torch.Tensor) -> np.ndarray:
        """Extract a 512-dimensional feature vector from a single crop tensor.

        Args:
            crop_tensor: Normalized tensor of shape (1, 3, 224, 224) or (3, 224, 224).

        Returns:
            1D NumPy array of shape (512,) and float32 dtype.
        """
        if self.model is None:
            raise RuntimeError("CNN model is not loaded.")

        if crop_tensor.ndim == 3:
            crop_tensor = crop_tensor.unsqueeze(0)

        with torch.no_grad():
            tensor = crop_tensor.to(self.device)
            features = self.model(tensor)
            return features.cpu().numpy().reshape(-1).astype(np.float32)

    def extract_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """Extract feature vectors from a batch of normalized crop tensors.

        Args:
            batch_tensor: Normalized tensor of shape (B, 3, 224, 224).

        Returns:
            2D NumPy array of shape (B, 512) and float32 dtype.
        """
        if self.model is None:
            raise RuntimeError("CNN model is not loaded.")

        if batch_tensor.numel() == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        with torch.no_grad():
            tensor = batch_tensor.to(self.device)
            features = self.model(tensor)
            return features.cpu().numpy().astype(np.float32)


def run_cnn_feature_extraction(
    video_id: str,
    frames_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    feature_extractor: CNNFeatureExtractor,
    behaviours_df: Optional[pd.DataFrame] = None,
    batch_size: int = DEFAULT_CNN_BATCH_SIZE,
    frame_selection_mode: str = "all",
    first_n: int = 10,
    range_start: int = 1,
    range_end: int = 18,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[bool, Optional[ExtractionSummary], Optional[pd.DataFrame], Optional[np.ndarray], str]:
    """Execute batched CNN visual feature extraction across tracked classroom video frames.

    Args:
        video_id: Unique identifier for the video.
        frames_df: DataFrame of extracted video frames metadata.
        tracks_df: DataFrame of Feature 4 tracking outputs.
        feature_extractor: Initialized CNNFeatureExtractor instance.
        behaviours_df: Optional DataFrame of Feature 5 observable behaviours.
        batch_size: Number of person crops to process per forward pass.
        frame_selection_mode: 'all', 'sample', or 'range'.
        first_n: Number of frames if mode is 'sample'.
        range_start: Start frame index if mode is 'range'.
        range_end: End frame index if mode is 'range'.
        progress_callback: Optional UI callback for progress reporting.

    Returns:
        Tuple of (success, summary, metadata_df, features_array, error_message).
    """
    start_time = time.time()

    if frames_df is None or frames_df.empty:
        return False, None, None, None, "Frames metadata is empty. Complete Feature 2 frame extraction first."

    if tracks_df is None or tracks_df.empty:
        return False, None, None, None, "Tracking dataset is empty. Complete Feature 4 tracking first."

    # Validate tracking columns
    req_track_cols = {"frame_id", "extracted_frame_index", "track_id", "x1", "y1", "x2", "y2"}
    missing_cols = req_track_cols - set(tracks_df.columns)
    if missing_cols:
        return False, None, None, None, f"Tracking data missing required columns: {missing_cols}"

    # Load behaviours.csv from disk if not explicitly passed
    if behaviours_df is None:
        beh_path = PROCESSED_DIR / video_id / "behaviours.csv"
        if beh_path.exists():
            try:
                behaviours_df = pd.read_csv(beh_path)
            except Exception:
                behaviours_df = None

    # Filter frames according to selection mode
    df_sorted = frames_df.sort_values("extracted_frame_index").copy()
    if frame_selection_mode == "sample":
        selected_frames = df_sorted.head(max(1, first_n))
    elif frame_selection_mode == "range":
        start_idx = max(1, range_start)
        end_idx = max(start_idx, range_end)
        selected_frames = df_sorted[
            (df_sorted["extracted_frame_index"] >= start_idx)
            & (df_sorted["extracted_frame_index"] <= end_idx)
        ]
    else:
        selected_frames = df_sorted

    if selected_frames.empty:
        return False, None, None, None, "No frames match the selected frame filtering criteria."

    total_frames = len(selected_frames)
    frame_indices = set(selected_frames["extracted_frame_index"].values)

    # Filter tracks corresponding to selected frames
    active_tracks = tracks_df[tracks_df["extracted_frame_index"].isin(frame_indices)].copy()
    if active_tracks.empty:
        return False, None, None, None, "No active tracks found in the selected frames."

    # Map behaviour predictions for fast O(1) lookup: (extracted_frame_index, track_id) -> (class, conf)
    beh_map: Dict[Tuple[int, int], Tuple[str, float]] = {}
    if behaviours_df is not None and not behaviours_df.empty:
        for _, row in behaviours_df.iterrows():
            f_idx = int(row.get("extracted_frame_index", -1))
            t_id = int(row.get("track_id", -1))
            b_cls = str(row.get("behaviour_class", "Unknown"))
            b_conf = float(row.get("confidence", 0.0))
            beh_map[(f_idx, t_id)] = (b_cls, b_conf)

    # Prepare extraction queues
    total_crops_evaluated = 0
    skipped_crops_count = 0
    feature_vectors_list: List[np.ndarray] = []
    metadata_rows: List[dict] = []

    # Sort tracks chronologically to preserve temporal order
    active_tracks = active_tracks.sort_values(["extracted_frame_index", "track_id"])

    # Batch accumulator
    batch_tensors: List[torch.Tensor] = []
    batch_metadata: List[dict] = []

    def flush_batch():
        nonlocal batch_tensors, batch_metadata
        if not batch_tensors:
            return
        stacked = torch.cat(batch_tensors, dim=0)
        features_np = feature_extractor.extract_batch(stacked)
        for i, feat in enumerate(features_np):
            feature_idx = len(feature_vectors_list)
            feature_vectors_list.append(feat)
            meta = batch_metadata[i]
            meta["feature_index"] = feature_idx
            metadata_rows.append(meta)
        batch_tensors = []
        batch_metadata = []

    # Group by frame for efficient sequential image loading
    frames_dir = FRAMES_DIR / video_id
    grouped_tracks = active_tracks.groupby("extracted_frame_index")
    frame_row_map = {row["extracted_frame_index"]: row for _, row in selected_frames.iterrows()}

    for f_step, (frame_idx, track_group) in enumerate(grouped_tracks, 1):
        if progress_callback:
            progress = min(1.0, f_step / max(1, total_frames))
            progress_callback(
                progress,
                f"Extracting CNN features for Frame {frame_idx} ({f_step}/{total_frames})...",
            )

        f_meta = frame_row_map.get(frame_idx)
        if f_meta is None:
            continue

        filename = str(f_meta.get("frame_filename", f"frame_{frame_idx:06d}.jpg"))
        frame_path = frames_dir / filename
        if not frame_path.exists():
            continue

        # Load frame image (OpenCV loads BGR, convert to RGB)
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frame_id_val = int(f_meta.get("frame_id", 0))
        timestamp_val = float(f_meta.get("timestamp_seconds", 0.0))

        for _, track in track_group.iterrows():
            total_crops_evaluated += 1
            bbox = (float(track["x1"]), float(track["y1"]), float(track["x2"]), float(track["y2"]))
            track_id_val = int(track["track_id"])

            success_crop, _, crop_tensor, _ = preprocess_person_crop(rgb_frame, bbox)
            if not success_crop or crop_tensor is None:
                skipped_crops_count += 1
                continue

            # Lookup linked behaviour if available
            beh_cls, beh_conf = beh_map.get((frame_idx, track_id_val), ("Unknown / Not Linked", 0.0))

            meta_entry = {
                "video_id": video_id,
                "frame_id": frame_id_val,
                "extracted_frame_index": frame_idx,
                "timestamp_seconds": round(timestamp_val, 3),
                "frame_filename": filename,
                "track_id": track_id_val,
                "confidence": round(float(track.get("confidence", 1.0)), 4),
                "x1": round(bbox[0], 2),
                "y1": round(bbox[1], 2),
                "x2": round(bbox[2], 2),
                "y2": round(bbox[3], 2),
                "feature_index": -1,  # will be assigned in flush_batch
                "feature_path": CNN_FEATURES_NPY_FILENAME,
                "behaviour_class": beh_cls,
                "behaviour_confidence": round(beh_conf, 4),
            }

            batch_tensors.append(crop_tensor)
            batch_metadata.append(meta_entry)

            if len(batch_tensors) >= batch_size:
                flush_batch()

    # Flush remaining batch items
    flush_batch()

    if not feature_vectors_list:
        return (
            False,
            None,
            None,
            None,
            f"No valid person crops could be extracted from {total_crops_evaluated} evaluated tracking boxes.",
        )

    # Combine feature vectors into (N, 512) array
    features_array = np.vstack(feature_vectors_list).astype(np.float32)
    metadata_df = pd.DataFrame(metadata_rows)

    # Save artifacts to data/processed/<video_id>/
    output_dir = PROCESSED_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / CNN_FEATURES_NPY_FILENAME
    csv_path = output_dir / CNN_METADATA_CSV_FILENAME

    np.save(str(npy_path), features_array)
    metadata_df.to_csv(csv_path, index=False)

    proc_time = time.time() - start_time

    summary = ExtractionSummary(
        video_id=video_id,
        model_name=feature_extractor.model_name,
        device=feature_extractor.device_name,
        feature_dimension=feature_extractor.feature_dim,
        total_frames_processed=total_frames,
        total_crops_evaluated=total_crops_evaluated,
        valid_features_extracted=len(features_array),
        skipped_crops_count=skipped_crops_count,
        processing_time_seconds=proc_time,
        features_npy_path=npy_path,
        metadata_csv_path=csv_path,
        status="Success",
    )

    if progress_callback:
        progress_callback(
            1.0,
            f"Feature extraction complete! {len(features_array)} vectors saved to {CNN_FEATURES_NPY_FILENAME}.",
        )

    return True, summary, metadata_df, features_array, ""
