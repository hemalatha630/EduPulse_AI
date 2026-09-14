"""Categorical Timeline and Trajectory Visualization Module for Feature 9.

Implements clear categorical visualizations for observable learning behaviour:
1. Categorical Gantt-style Timeline with tracking gaps and MM:SS timestamps.
2. Observed Behaviour Duration & Percentage Distribution bar chart.
3. Observable Behaviour Transition frequencies chart.
4. Model Comparison View (stacked rows comparing RNN, LSTM, and GRU side-by-side).
5. Classroom Multi-Track Overview (stacked horizontal tracks).

CRITICAL PEDAGOGICAL DESIGN RULE:
Observable behaviours are discrete categorical classes. They are NEVER plotted on
continuous numerical line charts to prevent implying numerical distance between classes.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behaviour.behaviour_labels import (
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
)
from src.config import BEHAVIOUR_COLORS
from src.trajectory.extractor import (
    BehaviourSegment,
    BehaviourTransition,
    TrackingGap,
    format_timestamp_mmss,
)


def get_behaviour_hex_color(behaviour_name: str) -> str:
    """Retrieve canonical hex color for an observable behaviour category."""
    if behaviour_name in BEHAVIOUR_COLORS:
        return BEHAVIOUR_COLORS[behaviour_name]["hex"]
    # Fallback to unknown gray
    return BEHAVIOUR_COLORS.get(CLASS_UNKNOWN, {}).get("hex", "#95A5A6")


def create_categorical_timeline_figure(
    segments: List[BehaviourSegment],
    gaps: Optional[List[TrackingGap]] = None,
    track_id: Optional[int] = None,
    model_name: str = "LSTM",
    time_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Generate a clean categorical Gantt-style horizontal timeline of observable behaviour.

    Args:
        segments: List of BehaviourSegment instances.
        gaps: Optional list of TrackingGap instances.
        track_id: Anonymous student track ID.
        model_name: Model name used for predictions.
        time_range: Optional tuple of (min_time, max_time) to bound the x-axis.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)

    # All target classes in consistent vertical order
    ordered_classes = list(TARGET_BEHAVIOUR_CLASSES) + [CLASS_UNKNOWN]
    y_map = {cls: idx for idx, cls in enumerate(ordered_classes)}

    # Plot tracking gaps first as background hatched spans
    if gaps:
        for gap in gaps:
            ax.axvspan(
                gap.start_timestamp_seconds,
                gap.end_timestamp_seconds,
                color="#ECEFF1",
                alpha=0.6,
                hatch="//",
                label="Tracking Gap" if "Tracking Gap" not in [p.get_label() for p in ax.patches] else "",
            )

    # Plot discrete categorical segment bars
    plotted_classes = set()
    for seg in segments:
        cls_name = seg.behaviour_class
        y_pos = y_map.get(cls_name, len(ordered_classes) - 1)
        bar_color = get_behaviour_hex_color(cls_name)

        # Draw segment horizontal bar
        rect = mpatches.Rectangle(
            xy=(seg.start_timestamp_seconds, y_pos - 0.35),
            width=max(0.1, seg.duration_seconds),
            height=0.7,
            facecolor=bar_color,
            edgecolor="#2C3E50",
            linewidth=1.2,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rect)
        plotted_classes.add(cls_name)

        # Label segment duration if wide enough
        if seg.duration_seconds >= 0.8:
            mid_x = seg.start_timestamp_seconds + seg.duration_seconds / 2.0
            ax.text(
                mid_x,
                y_pos,
                f"{seg.duration_seconds:.1f}s",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#FFFFFF" if cls_name != CLASS_UNKNOWN else "#2C3E50",
                zorder=4,
            )

    # Configure axes
    ax.set_yticks(range(len(ordered_classes)))
    ax.set_yticklabels(ordered_classes, fontsize=9.5, fontweight="medium")
    ax.set_ylim(-0.6, len(ordered_classes) - 0.4)

    # X-axis range and format
    if segments:
        min_x = min(s.start_timestamp_seconds for s in segments)
        max_x = max(s.end_timestamp_seconds for s in segments)
    else:
        min_x, max_x = 0.0, 10.0

    if time_range:
        min_x = min(min_x, time_range[0])
        max_x = max(max_x, time_range[1])

    pad = max(0.5, (max_x - min_x) * 0.03)
    ax.set_xlim(max(0.0, min_x - pad), max_x + pad)

    # Primary x-axis: video seconds, formatted with secondary MM:SS markers
    ax.set_xlabel("Elapsed Video Time (Seconds [MM:SS])", fontsize=10.5, fontweight="bold", labelpad=8)
    
    # Custom x-tick formatter to show seconds and MM:SS
    ticks = ax.get_xticks()
    valid_ticks = [t for t in ticks if t >= 0]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels([f"{int(t)}s\n({format_timestamp_mmss(t)})" for t in valid_ticks], fontsize=8.5)

    title_str = f"Observable Behaviour Trajectory — Track ID {track_id}" if track_id is not None else "Observable Behaviour Trajectory"
    ax.set_title(
        f"{title_str} (Model: {model_name.upper()})\n"
        f"Categorical timeline of observed actions across chronological sequence windows",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def create_duration_distribution_figure(
    durations_dict: Dict[str, float],
    percentages_dict: Dict[str, float],
    track_id: Optional[int] = None,
) -> plt.Figure:
    """Generate horizontal bar chart showing observed duration (seconds) and percentage of time.

    Args:
        durations_dict: Dict mapping class names to observed seconds.
        percentages_dict: Dict mapping class names to percentage (0..100).
        track_id: Anonymous student track ID.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)

    if not durations_dict:
        ax.text(0.5, 0.5, "No observed duration data available", ha="center", va="center")
        ax.axis("off")
        return fig

    # Sort descending by duration
    sorted_items = sorted(durations_dict.items(), key=lambda x: x[1])
    classes = [item[0] for item in sorted_items]
    durations = [item[1] for item in sorted_items]
    colors = [get_behaviour_hex_color(c) for c in classes]

    bars = ax.barh(classes, durations, color=colors, edgecolor="#2C3E50", height=0.6, alpha=0.9)

    # Data value labels on bars
    max_d = max(durations) if durations else 1.0
    for bar, cls_name in zip(bars, classes):
        w = bar.get_width()
        pct = percentages_dict.get(cls_name, 0.0)
        label_text = f" {w:.1f}s ({pct:.1f}%)"
        ax.text(
            w + (max_d * 0.015),
            bar.get_y() + bar.get_height() / 2.0,
            label_text,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#2C3E50",
        )

    ax.set_xlabel("Total Observed Duration (Seconds)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_xlim(0, max_d * 1.25)

    title_track = f"Track ID {track_id}" if track_id is not None else "Track"
    ax.set_title(
        f"Observed Behaviour Duration & Percentage of Time ({title_track})\n"
        "Calculated strictly from cumulative sequence observations",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def create_transitions_figure(
    transitions: List[BehaviourTransition],
    track_id: Optional[int] = None,
) -> plt.Figure:
    """Generate bar chart illustrating chronological behaviour transition counts.

    Args:
        transitions: List of BehaviourTransition objects.
        track_id: Anonymous student track ID.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=120)

    if not transitions:
        ax.text(
            0.5,
            0.5,
            "No behaviour transitions observed\n(Single sustained behaviour category throughout window)",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#546E7A",
        )
        ax.axis("off")
        return fig

    # Top 8 transitions for clean display
    display_trans = transitions[:8][::-1]
    labels = [t.label for t in display_trans]
    counts = [t.count for t in display_trans]

    bars = ax.barh(labels, counts, color="#3F51B5", edgecolor="#1A237E", height=0.55, alpha=0.85)

    max_c = max(counts) if counts else 1
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + max(0.1, max_c * 0.02),
            bar.get_y() + bar.get_height() / 2.0,
            f" {int(w)} transition{'s' if w > 1 else ''}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1A237E",
        )

    ax.set_xlabel("Transition Count", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_xlim(0, max_c * 1.35)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    title_track = f"Track ID {track_id}" if track_id is not None else "Track"
    ax.set_title(
        f"Observable Behaviour Transitions ({title_track})\n"
        "Direct transitions between consecutive distinct observable behaviours",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def create_model_comparison_timeline_figure(
    model_segments: Dict[str, List[BehaviourSegment]],
    track_id: Optional[int] = None,
    time_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Generate multi-row comparison timeline comparing RNN, LSTM, and GRU for the same track.

    Args:
        model_segments: Dict mapping model name ('RNN', 'LSTM', 'GRU') to BehaviourSegment list.
        track_id: Anonymous student track ID.
        time_range: Optional tuple of (min_time, max_time).

    Returns:
        Matplotlib Figure object.
    """
    models = list(model_segments.keys())
    fig, ax = plt.subplots(figsize=(12, 1.8 + 1.2 * len(models)), dpi=120)

    if not models:
        ax.text(0.5, 0.5, "No model comparison data available", ha="center", va="center")
        ax.axis("off")
        return fig

    y_positions = {m_name: idx for idx, m_name in enumerate(models)}

    all_start_times = []
    all_end_times = []

    for m_name, segs in model_segments.items():
        y_pos = y_positions[m_name]
        for seg in segs:
            all_start_times.append(seg.start_timestamp_seconds)
            all_end_times.append(seg.end_timestamp_seconds)

            color = get_behaviour_hex_color(seg.behaviour_class)
            rect = mpatches.Rectangle(
                xy=(seg.start_timestamp_seconds, y_pos - 0.35),
                width=max(0.1, seg.duration_seconds),
                height=0.7,
                facecolor=color,
                edgecolor="#2C3E50",
                linewidth=1.0,
                alpha=0.9,
                zorder=3,
            )
            ax.add_patch(rect)

            if seg.duration_seconds >= 0.8:
                mid_x = seg.start_timestamp_seconds + seg.duration_seconds / 2.0
                ax.text(
                    mid_x,
                    y_pos,
                    seg.behaviour_class[:16] + ("..." if len(seg.behaviour_class) > 16 else ""),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#FFFFFF" if seg.behaviour_class != CLASS_UNKNOWN else "#2C3E50",
                    zorder=4,
                )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m.upper()} Model" for m in models], fontsize=10, fontweight="bold")
    ax.set_ylim(-0.6, len(models) - 0.4)

    min_x = min(all_start_times) if all_start_times else 0.0
    max_x = max(all_end_times) if all_end_times else 10.0

    if time_range:
        min_x = min(min_x, time_range[0])
        max_x = max(max_x, time_range[1])

    pad = max(0.5, (max_x - min_x) * 0.03)
    ax.set_xlim(max(0.0, min_x - pad), max_x + pad)

    ax.set_xlabel("Elapsed Video Time (Seconds [MM:SS])", fontsize=10, fontweight="bold", labelpad=8)
    ticks = ax.get_xticks()
    valid_ticks = [t for t in ticks if t >= 0]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels([f"{int(t)}s\n({format_timestamp_mmss(t)})" for t in valid_ticks], fontsize=8.5)

    title_track = f"Track ID {track_id}" if track_id is not None else "Track"
    ax.set_title(
        f"Recurrent Architecture Comparison ({title_track})\n"
        "Side-by-side trajectory predictions generated by Vanilla RNN, LSTM, and GRU",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    # Legend for colours
    legend_patches = [
        mpatches.Patch(color=BEHAVIOUR_COLORS[c]["hex"], label=c)
        for c in TARGET_BEHAVIOUR_CLASSES
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        fontsize=8.5,
        frameon=True,
    )

    plt.tight_layout()
    return fig


def create_classroom_overview_figure(
    multi_track_segments: Dict[int, List[BehaviourSegment]],
    model_name: str = "LSTM",
    time_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Generate multi-track stacked timeline overview showing multiple anonymous students.

    Args:
        multi_track_segments: Dict mapping track_id (int) to its list of BehaviourSegments.
        model_name: Model used for predictions.
        time_range: Optional tuple of (min_time, max_time).

    Returns:
        Matplotlib Figure object.
    """
    sorted_tracks = sorted(multi_track_segments.keys())
    num_tracks = len(sorted_tracks)
    fig, ax = plt.subplots(figsize=(12, max(4.0, 1.0 + 0.45 * num_tracks)), dpi=120)

    if not sorted_tracks:
        ax.text(0.5, 0.5, "No multi-track trajectory data available", ha="center", va="center")
        ax.axis("off")
        return fig

    y_positions = {tid: idx for idx, tid in enumerate(sorted_tracks)}
    all_start_times = []
    all_end_times = []

    for tid, segs in multi_track_segments.items():
        y_pos = y_positions[tid]
        for seg in segs:
            all_start_times.append(seg.start_timestamp_seconds)
            all_end_times.append(seg.end_timestamp_seconds)

            color = get_behaviour_hex_color(seg.behaviour_class)
            rect = mpatches.Rectangle(
                xy=(seg.start_timestamp_seconds, y_pos - 0.35),
                width=max(0.1, seg.duration_seconds),
                height=0.7,
                facecolor=color,
                edgecolor="#37474F",
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
            )
            ax.add_patch(rect)

    ax.set_yticks(range(len(sorted_tracks)))
    ax.set_yticklabels([f"Track {tid}" for tid in sorted_tracks], fontsize=9, fontweight="medium")
    ax.set_ylim(-0.6, len(sorted_tracks) - 0.4)

    min_x = min(all_start_times) if all_start_times else 0.0
    max_x = max(all_end_times) if all_end_times else 10.0

    if time_range:
        min_x = min(min_x, time_range[0])
        max_x = max(max_x, time_range[1])

    pad = max(0.5, (max_x - min_x) * 0.03)
    ax.set_xlim(max(0.0, min_x - pad), max_x + pad)

    ax.set_xlabel("Elapsed Video Time (Seconds [MM:SS])", fontsize=10, fontweight="bold", labelpad=8)
    ticks = ax.get_xticks()
    valid_ticks = [t for t in ticks if t >= 0]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels([f"{int(t)}s\n({format_timestamp_mmss(t)})" for t in valid_ticks], fontsize=8.5)

    ax.set_title(
        f"Classroom Multi-Track Observable Behaviour Overview (Model: {model_name.upper()})\n"
        f"Timeline of observable behaviours across {num_tracks} anonymous tracked individuals",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=BEHAVIOUR_COLORS[c]["hex"], label=c)
        for c in TARGET_BEHAVIOUR_CLASSES
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=8.5,
        frameon=True,
    )

    plt.tight_layout()
    return fig
