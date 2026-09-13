"""Temporal Sequence Creation Module for EduPulse AI.

Organizes individual frame-level CNN visual features from Feature 6 into
chronologically ordered, fixed-length sliding-window temporal sequences
for each anonymous tracked student.
Outputs PyTorch-ready 3D tensors (N_sequences, sequence_length, feature_dimension)
and rich spatial-temporal metadata.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    CNN_FEATURES_NPY_FILENAME,
    CNN_METADATA_CSV_FILENAME,
    DEFAULT_MAX_FRAME_GAP,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEQUENCE_STRIDE,
    PROCESSED_DIR,
    TEMPORAL_SEQUENCES_METADATA_FILENAME,
    TEMPORAL_SEQUENCES_NPY_FILENAME,
)


@dataclass
class SequenceSummary:
    """Summary metrics of a temporal sequence creation session."""

    video_id: str
    total_tracks_evaluated: int
    valid_tracks_processed: int
    short_tracks_skipped: int
    total_sequences_created: int
    sequence_length: int
    stride: int
    feature_dimension: int
    tensor_shape: Tuple[int, int, int]
    avg_sequence_duration_seconds: float
    processing_time_seconds: float
    sequences_npy_path: Path
    metadata_csv_path: Path
    status: str

    def to_dict(self) -> dict:
        """Convert summary to serializable dictionary."""
        return {
            "video_id": self.video_id,
            "total_tracks_evaluated": self.total_tracks_evaluated,
            "valid_tracks_processed": self.valid_tracks_processed,
            "short_tracks_skipped": self.short_tracks_skipped,
            "total_sequences_created": self.total_sequences_created,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "feature_dimension": self.feature_dimension,
            "tensor_shape": list(self.tensor_shape),
            "avg_sequence_duration_seconds": round(self.avg_sequence_duration_seconds, 3),
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "sequences_npy_path": str(self.sequences_npy_path),
            "metadata_csv_path": str(self.metadata_csv_path),
            "status": self.status,
        }


class TemporalSequenceGenerator:
    """Configurable sliding-window temporal sequence generator.

    Groups CNN feature vectors strictly by track_id, sorts observations chronologically,
    splits continuous sequences across large tracking gaps, and applies sliding windows.
    """

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        stride: int = DEFAULT_SEQUENCE_STRIDE,
        max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    ):
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if max_frame_gap < 1:
            raise ValueError(f"max_frame_gap must be >= 1, got {max_frame_gap}")

        self.sequence_length = sequence_length
        self.stride = stride
        self.max_frame_gap = max_frame_gap

    def generate_track_sequences(
        self,
        track_df: pd.DataFrame,
        features_array: np.ndarray,
        video_id: str,
    ) -> Tuple[List[np.ndarray], List[dict], int]:
        """Generate sliding-window sequences for a single tracked student.

        Args:
            track_df: DataFrame of observations for a single track_id.
            features_array: Complete (N_total, D) feature array from Feature 6.
            video_id: Video identifier string.

        Returns:
            Tuple of:
            - sequence_arrays: List of (L, D) feature arrays.
            - sequence_metadata: List of metadata dicts.
            - skipped_segments_count: Number of track segments too short for sequence_length.
        """
        if track_df.empty:
            return [], [], 0

        # Sort observations chronologically by extracted frame index
        df_sorted = track_df.sort_values("extracted_frame_index").copy()
        track_id = int(df_sorted["track_id"].iloc[0])

        # Partition observations into continuous segments where gaps <= max_frame_gap
        segments: List[pd.DataFrame] = []
        current_segment_rows = [df_sorted.iloc[0]]

        for i in range(1, len(df_sorted)):
            prev_frame = int(df_sorted.iloc[i - 1]["extracted_frame_index"])
            curr_frame = int(df_sorted.iloc[i]["extracted_frame_index"])

            # Check if gap between consecutive observations exceeds tolerance
            if curr_frame - prev_frame > self.max_frame_gap:
                segments.append(pd.DataFrame(current_segment_rows))
                current_segment_rows = [df_sorted.iloc[i]]
            else:
                current_segment_rows.append(df_sorted.iloc[i])

        if current_segment_rows:
            segments.append(pd.DataFrame(current_segment_rows))

        sequences_list: List[np.ndarray] = []
        metadata_list: List[dict] = []
        skipped_short_count = 0

        for segment in segments:
            seg_len = len(segment)
            if seg_len < self.sequence_length:
                skipped_short_count += 1
                continue

            # Slide window across continuous segment
            for start_idx in range(0, seg_len - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                window = segment.iloc[start_idx:end_idx]

                feat_indices = window["feature_index"].astype(int).values
                window_features = features_array[feat_indices]  # Shape: (L, D)

                start_frame_id = int(window.iloc[0]["frame_id"])
                end_frame_id = int(window.iloc[-1]["frame_id"])
                start_ext = int(window.iloc[0]["extracted_frame_index"])
                end_ext = int(window.iloc[-1]["extracted_frame_index"])
                start_ts = float(window.iloc[0]["timestamp_seconds"])
                end_ts = float(window.iloc[-1]["timestamp_seconds"])
                duration = round(end_ts - start_ts, 3)

                # Behaviour labels alignment
                beh_classes = window.get("behaviour_class", pd.Series(["Unknown"] * len(window))).astype(str).tolist()
                beh_confs = window.get("behaviour_confidence", pd.Series([0.0] * len(window))).astype(float).tolist()

                # Calculate dominant observable behaviour (mode)
                valid_behs = [b for b in beh_classes if b != "Unknown" and b != "Unknown / Not Linked"]
                if valid_behs:
                    dominant_beh = max(set(valid_behs), key=valid_behs.count)
                else:
                    dominant_beh = "Unknown / Not Linked"

                beh_sequence_str = " -> ".join(beh_classes)
                mean_conf = round(float(np.mean(beh_confs)), 4) if beh_confs else 0.0

                frame_indices_list = window["extracted_frame_index"].astype(int).tolist()
                frame_timestamps_list = [round(float(ts), 3) for ts in window["timestamp_seconds"].tolist()]

                meta_entry = {
                    "sequence_id": -1,  # Assigned globally during final aggregation
                    "video_id": video_id,
                    "track_id": track_id,
                    "start_frame_id": start_frame_id,
                    "end_frame_id": end_frame_id,
                    "start_extracted_frame_index": start_ext,
                    "end_extracted_frame_index": end_ext,
                    "start_timestamp_seconds": round(start_ts, 3),
                    "end_timestamp_seconds": round(end_ts, 3),
                    "duration_seconds": duration,
                    "sequence_length": self.sequence_length,
                    "feature_dimension": int(window_features.shape[1]),
                    "dominant_behaviour": dominant_beh,
                    "mean_behaviour_confidence": mean_conf,
                    "behaviour_sequence": beh_sequence_str,
                    "frame_indices": json.dumps(frame_indices_list),
                    "frame_timestamps": json.dumps(frame_timestamps_list),
                }

                sequences_list.append(window_features)
                metadata_list.append(meta_entry)

        return sequences_list, metadata_list, skipped_short_count


def sequences_to_tensor(sequences_array: np.ndarray) -> torch.Tensor:
    """Convert 3D NumPy sequence array to PyTorch FloatTensor.

    Args:
        sequences_array: NumPy array of shape (N, L, D).

    Returns:
        PyTorch FloatTensor of shape (N, L, D).
    """
    if sequences_array is None or sequences_array.size == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)
    return torch.from_numpy(sequences_array).float()


class ClassroomSequenceDataset(Dataset):
    """PyTorch Dataset wrapper for classroom temporal visual sequences.

    Enables plug-and-play batching through standard PyTorch DataLoader for
    Feature 8 sequence models (RNN / LSTM / GRU).
    """

    def __init__(
        self,
        sequences: np.ndarray,
        metadata_df: Optional[pd.DataFrame] = None,
    ):
        self.sequences = sequences_to_tensor(sequences)
        self.metadata_df = metadata_df

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        sample = self.sequences[idx]
        meta = {}
        if self.metadata_df is not None and idx < len(self.metadata_df):
            meta = self.metadata_df.iloc[idx].to_dict()
        return sample, meta


def run_temporal_sequence_creation(
    video_id: str,
    cnn_features_path: Optional[Path] = None,
    cnn_metadata_path: Optional[Path] = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    stride: int = DEFAULT_SEQUENCE_STRIDE,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[bool, Optional[SequenceSummary], Optional[pd.DataFrame], Optional[np.ndarray], str]:
    """Execute temporal sequence creation from Feature 6 CNN visual features.

    Args:
        video_id: Unique identifier for the classroom video.
        cnn_features_path: Optional custom path to cnn_features.npy.
        cnn_metadata_path: Optional custom path to cnn_features_metadata.csv.
        sequence_length: Number of time steps per sequence window.
        stride: Stride step size between overlapping windows.
        max_frame_gap: Maximum allowed gap between consecutive observations in a track.
        progress_callback: Optional progress indicator callback.

    Returns:
        Tuple of (success, summary, metadata_df, sequences_array, error_message).
    """
    start_time = time.time()

    processed_dir = PROCESSED_DIR / video_id
    npy_path = cnn_features_path or (processed_dir / CNN_FEATURES_NPY_FILENAME)
    csv_path = cnn_metadata_path or (processed_dir / CNN_METADATA_CSV_FILENAME)

    if not npy_path.exists() or not csv_path.exists():
        return (
            False,
            None,
            None,
            None,
            "CNN features are required before temporal sequences can be created. "
            "Please complete Feature 6 first.",
        )

    try:
        features_array = np.load(str(npy_path))
    except Exception as exc:
        return False, None, None, None, f"Failed to load CNN features array: {exc}"

    try:
        meta_df = pd.read_csv(csv_path)
    except Exception as exc:
        return False, None, None, None, f"Failed to load CNN features metadata: {exc}"

    if features_array.ndim != 2 or len(features_array) == 0:
        return False, None, None, None, f"Invalid CNN feature array shape: {features_array.shape}."

    if meta_df.empty:
        return False, None, None, None, "CNN feature metadata table is empty."

    if len(features_array) != len(meta_df):
        return (
            False,
            None,
            None,
            None,
            f"Feature count mismatch: {len(features_array)} vectors in .npy vs {len(meta_df)} rows in .csv.",
        )

    # Initialize generator
    generator = TemporalSequenceGenerator(
        sequence_length=sequence_length,
        stride=stride,
        max_frame_gap=max_frame_gap,
    )

    # Group by track_id
    unique_tracks = sorted(meta_df["track_id"].unique())
    total_tracks = len(unique_tracks)
    all_sequence_arrays: List[np.ndarray] = []
    all_metadata_entries: List[dict] = []
    total_skipped_short = 0
    valid_tracks_count = 0

    for t_idx, track_id in enumerate(unique_tracks, 1):
        if progress_callback:
            prog = min(1.0, t_idx / max(1, total_tracks))
            progress_callback(prog, f"Processing Track ID {track_id} ({t_idx}/{total_tracks})...")

        track_rows = meta_df[meta_df["track_id"] == track_id]
        seqs, metas, skipped = generator.generate_track_sequences(
            track_df=track_rows,
            features_array=features_array,
            video_id=video_id,
        )

        total_skipped_short += skipped
        if seqs:
            valid_tracks_count += 1
            all_sequence_arrays.extend(seqs)
            all_metadata_entries.extend(metas)

    if not all_sequence_arrays:
        return (
            False,
            None,
            None,
            None,
            f"No temporal sequences could be created: all {total_tracks} tracks contained fewer "
            f"consecutive observations than the requested sequence length of {sequence_length} frames. "
            f"Please reduce the sequence length.",
        )

    # Assign sequential global sequence IDs (0 .. N-1)
    for seq_id, entry in enumerate(all_metadata_entries):
        entry["sequence_id"] = seq_id

    # Assemble 3D NumPy array: (N, L, D)
    sequences_tensor_np = np.stack(all_sequence_arrays, axis=0).astype(np.float32)
    sequences_metadata_df = pd.DataFrame(all_metadata_entries)

    # Save artifacts to data/processed/<video_id>/
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_npy_path = processed_dir / TEMPORAL_SEQUENCES_NPY_FILENAME
    out_csv_path = processed_dir / TEMPORAL_SEQUENCES_METADATA_FILENAME

    np.save(str(out_npy_path), sequences_tensor_np)
    sequences_metadata_df.to_csv(out_csv_path, index=False)

    proc_time = time.time() - start_time
    avg_duration = float(sequences_metadata_df["duration_seconds"].mean())

    summary = SequenceSummary(
        video_id=video_id,
        total_tracks_evaluated=total_tracks,
        valid_tracks_processed=valid_tracks_count,
        short_tracks_skipped=total_skipped_short,
        total_sequences_created=len(sequences_tensor_np),
        sequence_length=sequence_length,
        stride=stride,
        feature_dimension=int(sequences_tensor_np.shape[2]),
        tensor_shape=sequences_tensor_np.shape,
        avg_sequence_duration_seconds=avg_duration,
        processing_time_seconds=proc_time,
        sequences_npy_path=out_npy_path,
        metadata_csv_path=out_csv_path,
        status="Success",
    )

    if progress_callback:
        progress_callback(
            1.0,
            f"Temporal sequence creation complete! {len(sequences_tensor_np)} sequences saved.",
        )

    return True, summary, sequences_metadata_df, sequences_tensor_np, ""
