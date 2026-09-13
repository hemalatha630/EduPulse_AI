"""Temporal Sequence Visualization Module for EduPulse AI.

Provides sequence timeline plotting and track window coverage charts.
Strictly adheres to educational research boundaries: sequence visualization
reflects ordered temporal visual snapshots and does NOT represent an RNN/LSTM
model prediction or claim internal mental engagement.
"""

import json
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behaviour.behaviour_labels import get_behaviour_hex, get_behaviour_rgb
from src.tracking.tracker import get_track_color

TEMPORAL_DISCLAIMER_TITLE = "Chronological Observable Sequence Window"
TEMPORAL_DISCLAIMER_TEXT = (
    "Ordered sequence of visual feature snapshots. "
    "Does NOT represent an RNN/LSTM/GRU inference or claim internal mental engagement."
)


def create_sequence_timeline_figure(
    sequence_features: np.ndarray,
    sequence_meta: dict,
) -> plt.Figure:
    """Create a timeline figure illustrating an individual temporal sequence window.

    Plots feature vector L2 norm progression across time steps alongside aligned
    observable behaviour classifications.

    Args:
        sequence_features: Array of shape (L, D) containing visual embeddings.
        sequence_meta: Dictionary containing metadata for this sequence.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=100)
    fig.patch.set_facecolor("#111827")  # Slate dark background
    ax.set_facecolor("#1F2937")

    seq_len = len(sequence_features)
    time_steps = np.arange(1, seq_len + 1)

    # Compute L2 norm trajectory
    norms = np.linalg.norm(sequence_features, axis=1)

    # Parse frame indices and timestamps
    frame_indices = []
    if "frame_indices" in sequence_meta:
        try:
            val = sequence_meta["frame_indices"]
            frame_indices = json.loads(val) if isinstance(val, str) else list(val)
        except Exception:
            frame_indices = list(range(1, seq_len + 1))
    else:
        frame_indices = list(range(1, seq_len + 1))

    # Parse behaviour sequence
    behaviours = []
    if "behaviour_sequence" in sequence_meta and sequence_meta["behaviour_sequence"]:
        behaviours = [b.strip() for b in sequence_meta["behaviour_sequence"].split("->")]
    if len(behaviours) != seq_len:
        behaviours = [sequence_meta.get("dominant_behaviour", "Unknown")] * seq_len

    track_id = sequence_meta.get("track_id", 1)
    track_rgb = get_track_color(int(track_id))
    track_hex = f"#{track_rgb[0]:02x}{track_rgb[1]:02x}{track_rgb[2]:02x}"

    # Plot norm line
    ax.plot(
        time_steps,
        norms,
        color=track_hex,
        linewidth=2.5,
        alpha=0.85,
        linestyle="-",
        marker="o",
        markersize=7,
        label=f"Track ID {track_id} Embedding Norm",
    )

    # Annotate points with behaviour badges
    for idx, (step, norm_val, beh) in enumerate(zip(time_steps, norms, behaviours)):
        color_hex = get_behaviour_hex(beh)
        ax.scatter(step, norm_val, color=color_hex, s=70, zorder=5, edgecolors="white", linewidths=0.7)

    # Formatting
    start_ts = sequence_meta.get("start_timestamp_seconds", 0.0)
    end_ts = sequence_meta.get("end_timestamp_seconds", 0.0)
    seq_id = sequence_meta.get("sequence_id", 0)

    ax.set_title(
        f"{TEMPORAL_DISCLAIMER_TITLE} (Sequence #{seq_id} | Track ID {track_id})\n",
        color="#F9FAFB",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.text(
        0.5,
        0.91,
        TEMPORAL_DISCLAIMER_TEXT,
        ha="center",
        va="center",
        fontsize=8.0,
        fontstyle="italic",
        color="#9CA3AF",
    )

    ax.set_xlabel(
        f"Sequence Time Step (Frames {frame_indices[0]} → {frame_indices[-1]} | {start_ts:.2f}s → {end_ts:.2f}s)",
        color="#D1D5DB",
        fontsize=9.5,
    )
    ax.set_ylabel("CNN Feature Vector L2 Norm", color="#D1D5DB", fontsize=9.5)

    ax.set_xticks(time_steps)
    ax.set_xticklabels([f"t{i}\n(F{f})" for i, f in zip(time_steps, frame_indices)], color="#9CA3AF", fontsize=8)
    ax.tick_params(colors="#9CA3AF", which="both", labelsize=8.5)

    for spine in ax.spines.values():
        spine.set_color("#374151")

    ax.grid(True, linestyle="--", alpha=0.25, color="#4B5563")

    plt.tight_layout()
    return fig


def create_track_coverage_figure(metadata_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Create a Gantt-style chart showing temporal sequence windows across tracks.

    Args:
        metadata_df: DataFrame of temporal sequences metadata.

    Returns:
        Matplotlib Figure object or None if dataframe is empty.
    """
    if metadata_df.empty or "track_id" not in metadata_df.columns:
        return None

    unique_tracks = sorted(metadata_df["track_id"].unique())
    fig, ax = plt.subplots(figsize=(9.0, max(3.5, len(unique_tracks) * 0.45)), dpi=100)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1F2937")

    y_pos = {t_id: i for i, t_id in enumerate(unique_tracks)}

    for _, row in metadata_df.iterrows():
        t_id = int(row["track_id"])
        y = y_pos[t_id]
        start_t = float(row["start_timestamp_seconds"])
        duration = float(row["duration_seconds"])

        beh = row.get("dominant_behaviour", "Unknown")
        color_hex = get_behaviour_hex(beh)

        ax.barh(
            y,
            width=max(0.05, duration),
            left=start_t,
            height=0.45,
            align="center",
            color=color_hex,
            alpha=0.8,
            edgecolor="#111827",
            linewidth=0.5,
        )

    ax.set_yticks(range(len(unique_tracks)))
    ax.set_yticklabels([f"Track {t}" for t in unique_tracks], color="#D1D5DB", fontsize=9)
    ax.set_xlabel("Video Timeline (seconds)", color="#D1D5DB", fontsize=10)
    ax.set_title("Temporal Sequence Coverage by Track ID", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)

    ax.tick_params(colors="#9CA3AF", which="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#374151")

    ax.grid(True, linestyle="--", alpha=0.2, color="#4B5563", axis="x")

    plt.tight_layout()
    return fig
