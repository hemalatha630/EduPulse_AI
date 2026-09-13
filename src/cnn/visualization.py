"""Visual Feature Space Analysis and PCA Visualization Module for EduPulse AI.

Provides pure NumPy 2D Principal Component Analysis (PCA) projection and
matplotlib visual plotting for extracted CNN feature vectors.
Adheres strictly to scientific boundaries: visual similarity in embedding space
does NOT prove or infer mental engagement, cognitive focus, or internal emotional states.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behaviour.behaviour_labels import get_behaviour_hex, get_behaviour_rgb
from src.tracking.tracker import get_track_color

PCA_DISCLAIMER_TITLE = "2D PCA Visualization of CNN Features"
PCA_DISCLAIMER_TEXT = (
    "Visualizes similarity in extracted visual feature space only. "
    "Does NOT prove or infer internal mental engagement, motivation, or cognitive states."
)


def compute_pca_2d(features: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[float, float]]:
    """Compute 2D Principal Component Analysis (PCA) projection using NumPy SVD.

    Args:
        features: 2D NumPy array of shape (N, D), where D is feature dimension.

    Returns:
        Tuple of:
        - projected (Optional[np.ndarray]): Array of shape (N, 2) or None if N < 2.
        - explained_variance_ratio (Tuple[float, float]): Percentage of variance explained
          by PC1 and PC2.
    """
    if features is None or not isinstance(features, np.ndarray) or features.ndim != 2:
        return None, (0.0, 0.0)

    n_samples, n_features = features.shape
    if n_samples < 2 or n_features < 2:
        return None, (0.0, 0.0)

    try:
        # Center feature representations
        mean = np.mean(features, axis=0)
        centered = features - mean

        # SVD on centered data matrix
        u, s, vt = np.linalg.svd(centered, full_matrices=False)

        # 2D projection onto first 2 principal components
        k = min(2, len(s))
        projected = centered @ vt[:k].T

        # If only 1 singular value exists, pad second dimension with zeros
        if k == 1:
            projected = np.hstack([projected, np.zeros((n_samples, 1), dtype=projected.dtype)])
            var_explained = (1.0, 0.0)
        else:
            total_var = np.sum(s**2)
            if total_var > 0:
                ev1 = float((s[0] ** 2) / total_var)
                ev2 = float((s[1] ** 2) / total_var)
                var_explained = (ev1, ev2)
            else:
                var_explained = (0.0, 0.0)

        return projected, var_explained
    except Exception:
        return None, (0.0, 0.0)


def create_pca_scatter_figure(
    projected: np.ndarray,
    metadata_df: pd.DataFrame,
    color_by: str = "track_id",
    explained_variance: Tuple[float, float] = (0.0, 0.0),
) -> plt.Figure:
    """Create a high-contrast Matplotlib figure showing 2D PCA visual feature distribution.

    Args:
        projected: Array of shape (N, 2) containing 2D PCA coordinates.
        metadata_df: DataFrame of length N containing track_id and behaviour_class.
        color_by: Either 'track_id' or 'behaviour_class'.
        explained_variance: (ev_pc1, ev_pc2) variance ratios.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=100)
    fig.patch.set_facecolor("#111827")  # Slate dark background
    ax.set_facecolor("#1F2937")

    ev1_pct = explained_variance[0] * 100
    ev2_pct = explained_variance[1] * 100

    if color_by == "behaviour_class" and "behaviour_class" in metadata_df.columns:
        unique_groups = metadata_df["behaviour_class"].unique()
        for b_class in unique_groups:
            mask = metadata_df["behaviour_class"] == b_class
            pts = projected[mask]
            color_hex = get_behaviour_hex(b_class)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color_hex,
                label=str(b_class),
                alpha=0.85,
                s=45,
                edgecolors="white",
                linewidths=0.5,
            )
        legend_title = "Observable Behaviour"
    else:
        # Default: Color by Track ID
        unique_tracks = sorted(metadata_df["track_id"].unique())
        for t_id in unique_tracks:
            mask = metadata_df["track_id"] == t_id
            pts = projected[mask]
            rgb = get_track_color(int(t_id))
            color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color_hex,
                label=f"Track ID {t_id}",
                alpha=0.85,
                s=45,
                edgecolors="white",
                linewidths=0.5,
            )
        legend_title = "Tracked Students"

    ax.set_title(
        f"{PCA_DISCLAIMER_TITLE}\n",
        color="#F9FAFB",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    fig.text(
        0.5,
        0.92,
        PCA_DISCLAIMER_TEXT,
        ha="center",
        va="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#9CA3AF",
    )

    ax.set_xlabel(f"Principal Component 1 ({ev1_pct:.1f}% Variance)", color="#D1D5DB", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({ev2_pct:.1f}% Variance)", color="#D1D5DB", fontsize=10)

    ax.tick_params(colors="#9CA3AF", which="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#374151")

    ax.grid(True, linestyle="--", alpha=0.25, color="#4B5563")

    # Legend placement outside plot if many categories
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title=legend_title,
            title_fontsize=9,
            facecolor="#111827",
            edgecolor="#374151",
            labelcolor="#F3F4F6",
            fontsize=8,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

    plt.tight_layout()
    return fig
