"""Visual Analytics Module for Teaching Activity Analysis (Feature 10).

Implements publication-quality pedagogical visual analytics:
1. Horizontal Gantt-style Activity Timeline with MM:SS timecodes and dominant behaviour badges.
2. Grouped Bar Chart comparing observable behaviour percentage shares across teaching activities.
3. Activity-Behaviour Matrix Heatmap (Behaviours × Activities) displaying percentage shares.
4. Track-Level Activity Breakdown chart for individual student comparisons.

CRITICAL PEDAGOGICAL BOUNDARY:
All charts plot discrete observable behaviour categories across instructional modalities.
No continuous psychological constructs or internal mental states are inferred.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.activity.activity_labels import (
    ACTIVITY_UNKNOWN,
    TARGET_TEACHING_ACTIVITIES,
    get_activity_hex,
)
from src.activity.manager import TeachingActivitySegment
from src.behaviour.behaviour_labels import (
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
)
from src.config import BEHAVIOUR_COLORS
from src.trajectory.extractor import format_timestamp_mmss


def get_behaviour_hex_color(behaviour_name: str) -> str:
    """Retrieve canonical hex color string for an observable behaviour category."""
    if behaviour_name in BEHAVIOUR_COLORS:
        return BEHAVIOUR_COLORS[behaviour_name]["hex"]
    return BEHAVIOUR_COLORS.get(CLASS_UNKNOWN, {}).get("hex", "#95A5A6")


def create_activity_timeline_figure(
    segments: List[TeachingActivitySegment],
    video_duration_seconds: Optional[float] = None,
    summary_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Generate a clean horizontal Gantt-style timeline of instructional teaching activities.

    Args:
        segments: Chronologically sorted list of TeachingActivitySegment objects.
        video_duration_seconds: Total length of the video in seconds.
        summary_df: Optional summary DataFrame with 'Dominant Behaviour' column.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=120)

    if not segments:
        ax.text(
            0.5,
            0.5,
            "No teaching activity segments annotated.\nUse the annotation controls to define activity intervals.",
            ha="center",
            va="center",
            fontsize=12,
            color="#7F8C8D",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    # Total duration
    max_t = max(s.end_timestamp_seconds for s in segments)
    if video_duration_seconds is not None and video_duration_seconds > max_t:
        max_t = video_duration_seconds
    max_t = max(1.0, max_t)

    # Activity dominant behaviour lookup map
    dom_map: Dict[str, str] = {}
    if summary_df is not None and not summary_df.empty and "Activity" in summary_df.columns:
        for _, row in summary_df.iterrows():
            dom_map[str(row["Activity"])] = str(row.get("Dominant Behaviour", ""))

    y_bar_bottom = 0.35
    bar_height = 0.40

    for s in segments:
        start_t = s.start_timestamp_seconds
        dur = max(0.01, s.duration_seconds)
        color = get_activity_hex(s.activity_class)

        # Draw segment bar
        rect = mpatches.Rectangle(
            (start_t, y_bar_bottom),
            dur,
            bar_height,
            facecolor=color,
            edgecolor="#2C3E50",
            linewidth=1.5,
            alpha=0.92,
        )
        ax.add_patch(rect)

        # Center text label inside or above bar
        mid_t = start_t + (dur / 2.0)
        timecode_str = f"{s.start_time_formatted} - {s.end_time_formatted}"

        # Adjust text depending on bar width
        if (dur / max_t) > 0.15:
            ax.text(
                mid_t,
                y_bar_bottom + (bar_height / 2.0) + 0.05,
                s.activity_class,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
            ax.text(
                mid_t,
                y_bar_bottom + (bar_height / 2.0) - 0.08,
                timecode_str,
                ha="center",
                va="center",
                fontsize=9,
                color="#ECF0F1",
            )
        elif (dur / max_t) > 0.08:
            ax.text(
                mid_t,
                y_bar_bottom + (bar_height / 2.0),
                s.activity_class,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
        else:
            # Short segment: write above bar
            ax.text(
                mid_t,
                y_bar_bottom + bar_height + 0.04,
                s.activity_class,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#2C3E50",
                rotation=45,
            )

        # Dominant behaviour badge below bar
        dom_b = dom_map.get(s.activity_class, "")
        if dom_b and dom_b != "None (No observations)" and (dur / max_t) > 0.18:
            # Shorten label if long
            short_dom = dom_b.split("(")[0].strip()
            ax.text(
                mid_t,
                y_bar_bottom - 0.08,
                f"Most observed:\n{short_dom}",
                ha="center",
                va="top",
                fontsize=8,
                style="italic",
                color="#34495E",
            )

    ax.set_xlim(0, max_t)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])

    # Custom timecode ticks on X-axis
    num_ticks = min(8, max(4, int(max_t // 15) + 1))
    tick_locs = np.linspace(0, max_t, num_ticks)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([format_timestamp_mmss(t) for t in tick_locs], fontsize=10)
    ax.set_xlabel("Elapsed Classroom Time (MM:SS)", fontsize=11, fontweight="bold", labelpad=8)

    ax.set_title(
        "Instructional Teaching Activity Timeline & Segmentation",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#2C3E50",
    )

    # Custom legend for teaching activities
    unique_acts = []
    seen = set()
    for s in segments:
        if s.activity_class not in seen:
            unique_acts.append(s.activity_class)
            seen.add(s.activity_class)

    legend_patches = [
        mpatches.Patch(facecolor=get_activity_hex(a), edgecolor="#2C3E50", label=a)
        for a in unique_acts
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.28),
            ncol=min(4, len(legend_patches)),
            frameon=True,
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#BDC3C7")

    fig.tight_layout()
    return fig


def create_activity_behaviour_distribution_figure(
    distribution_df: pd.DataFrame,
    title: str = "Classroom Observable Behaviour Distribution by Teaching Activity",
) -> plt.Figure:
    """Generate a grouped bar chart comparing observable behaviour percentage shares across activities.

    Args:
        distribution_df: DataFrame returned by calculate_activity_behaviour_distributions.
        title: Chart title string.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)

    if distribution_df.empty:
        ax.text(
            0.5,
            0.5,
            "No behaviour observations available for the designated activities.",
            ha="center",
            va="center",
            fontsize=12,
            color="#7F8C8D",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    activities = [
        a for a in TARGET_TEACHING_ACTIVITIES if a in distribution_df["activity_class"].unique()
    ]
    if not activities:
        activities = distribution_df["activity_class"].unique().tolist()

    behaviours = [
        b for b in TARGET_BEHAVIOUR_CLASSES if b in distribution_df["behaviour_class"].unique()
    ]
    if not behaviours:
        behaviours = distribution_df["behaviour_class"].unique().tolist()

    n_activities = len(activities)
    n_behaviours = len(behaviours)

    if n_activities == 0 or n_behaviours == 0:
        ax.text(0.5, 0.5, "Insufficient data to plot.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    x = np.arange(n_activities)
    bar_width = 0.75 / max(1, n_behaviours)

    for i, b_cls in enumerate(behaviours):
        b_color = get_behaviour_hex_color(b_cls)
        percentages = []
        for act in activities:
            sub = distribution_df[
                (distribution_df["activity_class"] == act)
                & (distribution_df["behaviour_class"] == b_cls)
            ]
            pct = float(sub["percentage_share"].iloc[0]) if not sub.empty else 0.0
            percentages.append(pct)

        offsets = x - (0.75 / 2.0) + (i + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            percentages,
            width=bar_width,
            label=b_cls,
            color=b_color,
            edgecolor="#2C3E50",
            linewidth=0.8,
            alpha=0.9,
        )

        # Label non-zero percentages on top of bars
        for bar, pct in zip(bars, percentages):
            if pct >= 4.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.8,
                    f"{pct:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#2C3E50",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(activities, fontsize=11, fontweight="bold")
    ax.set_ylabel("Share of Observed Time (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylim(0, 105)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14, color="#2C3E50")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.25),
        ncol=min(3, n_behaviours),
        frameon=True,
        fontsize=9,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#BDC3C7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDC3C7")
    ax.spines["bottom"].set_color("#BDC3C7")

    fig.tight_layout()
    return fig


def create_activity_behaviour_heatmap_figure(
    distribution_df: pd.DataFrame,
) -> plt.Figure:
    """Generate an activity-behaviour matrix heatmap displaying percentage shares.

    Rows: Observable Behaviour Classes
    Columns: Teaching Activities
    Cells: Share of observed time in that activity (%)

    Args:
        distribution_df: DataFrame returned by calculate_activity_behaviour_distributions.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)

    if distribution_df.empty:
        ax.text(
            0.5,
            0.5,
            "No distribution data available for heatmap generation.",
            ha="center",
            va="center",
            fontsize=12,
            color="#7F8C8D",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    activities = [
        a for a in TARGET_TEACHING_ACTIVITIES if a in distribution_df["activity_class"].unique()
    ]
    if not activities:
        activities = distribution_df["activity_class"].unique().tolist()

    behaviours = [
        b for b in TARGET_BEHAVIOUR_CLASSES if b in distribution_df["behaviour_class"].unique()
    ]
    if not behaviours:
        behaviours = distribution_df["behaviour_class"].unique().tolist()

    # Build 2D matrix
    matrix = np.zeros((len(behaviours), len(activities)), dtype=float)
    for r_idx, b_cls in enumerate(behaviours):
        for c_idx, act in enumerate(activities):
            sub = distribution_df[
                (distribution_df["activity_class"] == act)
                & (distribution_df["behaviour_class"] == b_cls)
            ]
            pct = float(sub["percentage_share"].iloc[0]) if not sub.empty else 0.0
            matrix[r_idx, c_idx] = pct

    # Color map
    cmap = plt.cm.YlGnBu
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Add text annotations
    for r in range(len(behaviours)):
        for c in range(len(activities)):
            val = matrix[r, c]
            text_color = "white" if val > 55 else "#2C3E50"
            ax.text(
                c,
                r,
                f"{val:.1f}%\n({distribution_df[(distribution_df['activity_class'] == activities[c]) & (distribution_df['behaviour_class'] == behaviours[r])]['observed_duration_seconds'].iloc[0]:.1f}s)"
                if not distribution_df[
                    (distribution_df["activity_class"] == activities[c])
                    & (distribution_df["behaviour_class"] == behaviours[r])
                ].empty
                else f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(np.arange(len(activities)))
    ax.set_xticklabels(activities, fontsize=11, fontweight="bold")
    ax.set_yticks(np.arange(len(behaviours)))
    ax.set_yticklabels(behaviours, fontsize=10, fontweight="bold")

    ax.set_title(
        "Activity × Behaviour Cross-Tabulation Matrix (Observed Share %)",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#2C3E50",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of Activity Observed Time (%)", fontsize=10, fontweight="bold", labelpad=8)

    fig.tight_layout()
    return fig


def create_track_activity_figure(
    track_distribution_df: pd.DataFrame,
    track_id: int,
) -> plt.Figure:
    """Generate a track-level activity-wise behaviour breakdown chart.

    Args:
        track_distribution_df: DataFrame returned by calculate_activity_behaviour_distributions
            filtered for track_id.
        track_id: Integer identifier of anonymous student track.

    Returns:
        Matplotlib Figure object.
    """
    title = f"Student Track #{track_id}: Observable Behaviour Breakdown Across Teaching Activities"
    return create_activity_behaviour_distribution_figure(
        distribution_df=track_distribution_df,
        title=title,
    )
