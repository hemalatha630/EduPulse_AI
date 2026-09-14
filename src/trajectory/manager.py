"""Trajectory Prediction Management and Export Module for Feature 9.

Provides:
1. Zero-retraining inference reuse: directly loads trained Feature 8 checkpoints (RNN, LSTM, GRU).
2. Trajectory prediction caching to disk (results/trajectories/<video_id>/).
3. Structured export of behaviour trajectories with segment metadata to CSV.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.config import (
    CNN_FEATURE_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
    MODELS_TEMPORAL_DIR,
    PROCESSED_DIR,
    RESULTS_TRAJECTORIES_DIR,
    SUPPORTED_TEMPORAL_MODELS,
    TEMPORAL_SEQUENCES_METADATA_FILENAME,
    TEMPORAL_SEQUENCES_NPY_FILENAME,
    TRAJECTORIES_CSV_FILENAME,
)
from src.temporal.dataset import IDX_TO_CLASS, NUM_TARGET_CLASSES
from src.temporal.models import BaseTemporalClassifier, create_temporal_model
from src.trajectory.extractor import merge_behaviour_segments


def load_or_generate_trajectory_predictions(
    video_id: str,
    model_type: str = MODEL_LSTM,
    processed_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    trajectories_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Load or generate full-video sequence predictions using an existing trained model.

    STRICT ZERO-RETRAINING RULE:
    This function NEVER trains a neural network. It reuses existing saved predictions
    or executes a fast forward pass using the saved Feature 8 model checkpoint.

    Args:
        video_id: Video identifier stem (e.g. 'classroom_lecture_demo').
        model_type: Recurrent model architecture ('rnn', 'lstm', 'gru').
        processed_dir: Directory containing Feature 7 sequence artifacts.
        models_dir: Directory containing Feature 8 model checkpoints.
        trajectories_dir: Directory to cache trajectory predictions.
        device: Torch device (CPU or CUDA).
        force_regenerate: If True, regenerates from checkpoint even if cached CSV exists.

    Returns:
        DataFrame containing sequence-level predictions for all tracks.
    """
    m_type = model_type.lower()
    if m_type not in SUPPORTED_TEMPORAL_MODELS:
        raise ValueError(
            f"Unsupported temporal model '{model_type}'. Supported: {SUPPORTED_TEMPORAL_MODELS}"
        )

    out_traj_dir = (trajectories_dir or RESULTS_TRAJECTORIES_DIR) / video_id
    out_traj_dir.mkdir(parents=True, exist_ok=True)
    cached_csv_path = out_traj_dir / f"predictions_{m_type}.csv"

    # Return cached predictions if available
    if not force_regenerate and cached_csv_path.exists():
        try:
            df = pd.read_csv(cached_csv_path)
            if not df.empty and "track_id" in df.columns:
                return df
        except Exception:
            pass

    # Verify Feature 7 prerequisites
    p_dir = (processed_dir or PROCESSED_DIR) / video_id
    seq_npy_path = p_dir / TEMPORAL_SEQUENCES_NPY_FILENAME
    seq_meta_path = p_dir / TEMPORAL_SEQUENCES_METADATA_FILENAME

    if not (seq_npy_path.exists() and seq_meta_path.exists()):
        raise FileNotFoundError(
            f"Temporal sequence files not found in {p_dir}. "
            "Please complete Feature 7 (Temporal Sequence Creation) first."
        )

    # Verify Feature 8 model checkpoint
    m_dir = models_dir or MODELS_TEMPORAL_DIR
    ckpt_path = m_dir / f"{m_type}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint '{ckpt_path}' not found. "
            f"Please train the {m_type.upper()} model in Feature 8 first."
        )

    seq_array = np.load(seq_npy_path)
    seq_meta_df = pd.read_csv(seq_meta_path)

    if len(seq_array) == 0 or len(seq_meta_df) == 0:
        return pd.DataFrame()

    # Load model architecture from checkpoint
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=dev)

    config_dict = ckpt.get("config", {})
    hidden_size = config_dict.get("hidden_size", DEFAULT_HIDDEN_SIZE)
    num_layers = config_dict.get("num_layers", DEFAULT_NUM_LAYERS)
    dropout = config_dict.get("dropout", DEFAULT_DROPOUT)
    input_size = seq_array.shape[2] if seq_array.ndim == 3 else CNN_FEATURE_DIM

    model = create_temporal_model(
        model_type=m_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=NUM_TARGET_CLASSES,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    # Batched forward pass without gradient computation
    seq_tensor = torch.from_numpy(seq_array).float().to(dev)
    batch_size = 32
    all_preds = []
    all_confs = []

    with torch.no_grad():
        for start_idx in range(0, len(seq_tensor), batch_size):
            batch_x = seq_tensor[start_idx : start_idx + batch_size]
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    # Assemble prediction records
    records = []
    for i in range(len(seq_meta_df)):
        meta_row = seq_meta_df.iloc[i]
        pred_idx = all_preds[i]
        conf_val = all_confs[i]
        pred_label = IDX_TO_CLASS.get(pred_idx, "Unknown")

        records.append(
            {
                "model": m_type.upper(),
                "video_id": video_id,
                "track_id": int(meta_row.get("track_id", 0)),
                "sequence_id": int(meta_row.get("sequence_id", i)),
                "start_frame_id": int(meta_row.get("start_frame_id", meta_row.get("start_extracted_frame_index", 0))),
                "end_frame_id": int(meta_row.get("end_frame_id", meta_row.get("end_extracted_frame_index", 0))),
                "start_timestamp_seconds": round(float(meta_row.get("start_timestamp_seconds", 0.0)), 3),
                "end_timestamp_seconds": round(float(meta_row.get("end_timestamp_seconds", 0.0)), 3),
                "duration_seconds": round(float(meta_row.get("duration_seconds", 0.0)), 3),
                "ground_truth_behaviour": str(meta_row.get("dominant_behaviour", "Unknown")),
                "predicted_label": pred_label,
                "confidence": round(float(conf_val), 4),
            }
        )

    predictions_df = pd.DataFrame(records)
    predictions_df.to_csv(cached_csv_path, index=False)
    return predictions_df


def export_behaviour_trajectories_csv(
    video_id: str,
    predictions_df: pd.DataFrame,
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Export complete trajectory dataset combining sequence-level and segment-level data.

    Args:
        video_id: Video stem.
        predictions_df: Sequence predictions DataFrame.
        output_csv_path: Optional output file path.

    Returns:
        Consolidated trajectory DataFrame saved to disk.
    """
    if predictions_df.empty:
        return pd.DataFrame()

    enriched_records = []
    # Process each track individually to map segment IDs and boundaries
    for tid, track_group in predictions_df.groupby("track_id"):
        track_sorted = track_group.sort_values(by="start_timestamp_seconds").reset_index(drop=True)
        segments = merge_behaviour_segments(track_sorted)

        # Map each sequence to its enclosing segment
        for _, seq_row in track_sorted.iterrows():
            seq_id = int(seq_row["sequence_id"])
            matching_seg = next(
                (s for s in segments if s.start_sequence_id <= seq_id <= s.end_sequence_id),
                None,
            )

            rec = seq_row.to_dict()
            if matching_seg:
                rec["segment_id"] = matching_seg.segment_id
                rec["segment_start_timestamp_seconds"] = matching_seg.start_timestamp_seconds
                rec["segment_end_timestamp_seconds"] = matching_seg.end_timestamp_seconds
                rec["segment_duration_seconds"] = matching_seg.duration_seconds
            else:
                rec["segment_id"] = 0
                rec["segment_start_timestamp_seconds"] = rec["start_timestamp_seconds"]
                rec["segment_end_timestamp_seconds"] = rec["end_timestamp_seconds"]
                rec["segment_duration_seconds"] = rec["duration_seconds"]

            enriched_records.append(rec)

    enriched_df = pd.DataFrame(enriched_records)

    out_file = output_csv_path or (
        RESULTS_TRAJECTORIES_DIR / video_id / TRAJECTORIES_CSV_FILENAME
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(out_file, index=False)
    return enriched_df
