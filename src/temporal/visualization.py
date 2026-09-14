"""Temporal Sequence Visualization Module for EduPulse AI.

Provides sequence timeline plotting and track window coverage charts.
Strictly adheres to educational research boundaries: sequence visualization
reflects ordered temporal visual snapshots and does NOT represent an RNN/LSTM
model prediction or claim internal mental engagement.
"""

import json
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
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


def create_training_curves_figure(
    history_df: pd.DataFrame,
    model_name: str,
) -> Optional[plt.Figure]:
    """Create a dual-panel figure showing training & validation loss and accuracy curves.

    Args:
        history_df: DataFrame with columns: epoch, train_loss, train_acc, val_loss, val_acc.
        model_name: Name of model (e.g. 'RNN', 'LSTM', 'GRU').

    Returns:
        Matplotlib Figure object or None if history is empty.
    """
    if history_df.empty or "epoch" not in history_df.columns:
        return None

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=100)
    fig.patch.set_facecolor("#111827")

    epochs = history_df["epoch"].values

    # Left Panel: Loss
    ax_loss.set_facecolor("#1F2937")
    ax_loss.plot(
        epochs,
        history_df["train_loss"],
        color="#38BDF8",  # Sky blue
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Train Loss",
    )
    ax_loss.plot(
        epochs,
        history_df["val_loss"],
        color="#F43F5E",  # Rose red
        linewidth=2.2,
        marker="s",
        markersize=4.5,
        linestyle="--",
        label="Val Loss",
    )
    ax_loss.set_title(f"{model_name.upper()} — Loss Curve", color="#F9FAFB", fontsize=11, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", color="#D1D5DB", fontsize=9.5)
    ax_loss.set_ylabel("CrossEntropy Loss", color="#D1D5DB", fontsize=9.5)
    ax_loss.tick_params(colors="#9CA3AF", labelsize=8.5)
    ax_loss.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8.5)
    ax_loss.grid(True, linestyle="--", alpha=0.25, color="#4B5563")
    for spine in ax_loss.spines.values():
        spine.set_color("#374151")

    # Right Panel: Accuracy
    ax_acc.set_facecolor("#1F2937")
    ax_acc.plot(
        epochs,
        history_df["train_acc"] * 100,
        color="#34D399",  # Emerald green
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Train Accuracy (%)",
    )
    ax_acc.plot(
        epochs,
        history_df["val_acc"] * 100,
        color="#FBBF24",  # Amber
        linewidth=2.2,
        marker="s",
        markersize=4.5,
        linestyle="--",
        label="Val Accuracy (%)",
    )
    ax_acc.set_title(f"{model_name.upper()} — Accuracy Curve", color="#F9FAFB", fontsize=11, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch", color="#D1D5DB", fontsize=9.5)
    ax_acc.set_ylabel("Accuracy (%)", color="#D1D5DB", fontsize=9.5)
    ax_acc.tick_params(colors="#9CA3AF", labelsize=8.5)
    ax_acc.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8.5)
    ax_acc.grid(True, linestyle="--", alpha=0.25, color="#4B5563")
    for spine in ax_acc.spines.values():
        spine.set_color("#374151")

    plt.tight_layout()
    return fig


def create_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
) -> plt.Figure:
    """Create an annotated confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (K, K).
        class_names: List of K class name strings.
        model_name: Name of model for title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=100)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1F2937")

    # Abbreviate long class names for clean axis display
    short_names = [
        name.replace("Looking toward the instructional activity", "Instruction")
        .replace("Reading/writing", "Read/Write")
        .replace("Interacting with peers", "Peer Inter.")
        .replace("Looking away", "Look Away")
        .replace("Mobile-device activity", "Mobile")
        .replace("Head-down behaviour", "Head Down")
        for name in class_names
    ]

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#9CA3AF", labelsize=8.5)

    num_classes = len(class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            color = "#FFFFFF" if val > thresh else "#111827" if val > 0 else "#6B7280"
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                fontweight="bold" if val > 0 else "normal",
            )

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(short_names, rotation=35, ha="right", color="#D1D5DB", fontsize=9)
    ax.set_yticklabels(short_names, color="#D1D5DB", fontsize=9)

    ax.set_xlabel("Predicted Observable Behaviour", color="#F9FAFB", fontsize=10, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Observable Behaviour", color="#F9FAFB", fontsize=10, fontweight="bold", labelpad=10)

    ax.set_title(
        f"{model_name.upper()} — Test Set Confusion Matrix\n",
        color="#F9FAFB",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    fig.text(
        0.5,
        0.93,
        "Observable behaviour classification errors only. Does NOT reflect internal mental states.",
        ha="center",
        va="center",
        fontsize=8.0,
        fontstyle="italic",
        color="#9CA3AF",
    )

    for spine in ax.spines.values():
        spine.set_color("#374151")

    plt.tight_layout()
    return fig


def create_model_comparison_figure(
    comparison_df: pd.DataFrame,
) -> Optional[plt.Figure]:
    """Create a grouped bar chart comparing Accuracy, Macro F1, and Weighted F1 across models.

    Args:
        comparison_df: DataFrame with columns: Model, Accuracy, Macro F1, Weighted F1.

    Returns:
        Matplotlib Figure object or None if dataframe is empty.
    """
    if comparison_df.empty or "Model" not in comparison_df.columns:
        return None

    models = comparison_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    # Helper to parse percentage or float strings
    def parse_col(col_name: str) -> List[float]:
        vals = []
        for v in comparison_df[col_name]:
            v_str = str(v).replace("%", "").strip()
            try:
                val_f = float(v_str)
                vals.append(val_f / 100.0 if "%" in str(v) else val_f)
            except Exception:
                vals.append(0.0)
        return vals

    accs = parse_col("Accuracy")
    macro_f1s = parse_col("Macro F1")
    weighted_f1s = parse_col("Weighted F1")

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=100)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1F2937")

    rects1 = ax.bar(x - width, [a * 100 for a in accs], width, label="Accuracy (%)", color="#38BDF8", alpha=0.9)
    rects2 = ax.bar(x, [f * 100 for f in macro_f1s], width, label="Macro F1 (×100)", color="#A855F7", alpha=0.9)
    rects3 = ax.bar(x + width, [w * 100 for w in weighted_f1s], width, label="Weighted F1 (×100)", color="#34D399", alpha=0.9)

    ax.set_ylabel("Score (%)", color="#D1D5DB", fontsize=9.5)
    ax.set_title("Fair Model Comparison — RNN vs LSTM vs GRU", color="#F9FAFB", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="#F9FAFB", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#9CA3AF", labelsize=8.5)
    ax.set_ylim(0, 110)
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.2, color="#4B5563", axis="y")

    # Value labels on top of bars
    for rect in list(rects1) + list(rects2) + list(rects3):
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#E5E7EB",
        )

    for spine in ax.spines.values():
        spine.set_color("#374151")

    plt.tight_layout()
    return fig
