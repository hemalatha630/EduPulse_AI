"""Teaching Activity Analysis and Behaviour Distribution Engine for EduPulse AI.

Provides:
1. Mapping recurrent observable behaviour predictions to teaching activity intervals.
2. Activity-wise behaviour distribution analysis (duration in seconds and percentage share).
3. Classroom-level and track-level breakdown.
4. Summary generation with explicit tie detection.
5. Activity-specific behaviour transition analysis.
6. Export of empirical activity analysis artifacts.

Academic Scope:
All metrics characterize observable student behaviors across instructional classroom activities.
No inferences regarding internal mental, cognitive, or emotional states are made.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.activity.activity_labels import (
    ACTIVITY_UNKNOWN,
    TARGET_TEACHING_ACTIVITIES,
)
from src.activity.manager import TeachingActivitySegment
from src.behaviour.behaviour_labels import (
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
)
from src.config import (
    ACTIVITY_BEHAVIOUR_SUMMARY_FILENAME,
    ACTIVITY_TRANSITION_SUMMARY_FILENAME,
    RESULTS_ACTIVITY_DIR,
)
from src.trajectory.extractor import (
    BehaviourSegment,
    BehaviourTransition,
    calculate_behaviour_transitions,
    format_timestamp_mmss,
    merge_behaviour_segments,
)


def map_predictions_to_activities(
    predictions_df: pd.DataFrame,
    segments: List[TeachingActivitySegment],
) -> pd.DataFrame:
    """Assign teaching activity segment labels to observable behaviour predictions.

    Maps each prediction record to its corresponding instructional activity segment
    based on temporal occurrence.

    Args:
        predictions_df: DataFrame containing behaviour predictions with timestamp columns.
        segments: List of validated TeachingActivitySegment objects.

    Returns:
        Enriched DataFrame with columns:
        'activity_class', 'segment_id', 'annotation_source'.
    """
    if predictions_df.empty:
        df_copy = predictions_df.copy()
        df_copy["activity_class"] = pd.Series(dtype="str")
        df_copy["segment_id"] = pd.Series(dtype="int")
        df_copy["annotation_source"] = pd.Series(dtype="str")
        return df_copy

    df_copy = predictions_df.copy()

    # Determine representative timestamp for each prediction row
    if "start_timestamp_seconds" in df_copy.columns and "end_timestamp_seconds" in df_copy.columns:
        rep_timestamps = (
            df_copy["start_timestamp_seconds"] + df_copy["end_timestamp_seconds"]
        ) / 2.0
    elif "timestamp_seconds" in df_copy.columns:
        rep_timestamps = df_copy["timestamp_seconds"]
    elif "start_timestamp" in df_copy.columns:
        rep_timestamps = df_copy["start_timestamp"]
    else:
        rep_timestamps = pd.Series(0.0, index=df_copy.index)

    activity_classes: List[str] = []
    segment_ids: List[int] = []
    sources: List[str] = []

    sorted_segs = sorted(segments, key=lambda s: s.start_timestamp_seconds)

    for t in rep_timestamps:
        matched_seg = None
        for i, s in enumerate(sorted_segs):
            is_last = i == (len(sorted_segs) - 1)
            # Standard interval [start, end) or inclusive [start, end] on final segment
            if s.start_timestamp_seconds <= t < s.end_timestamp_seconds or (
                is_last and s.start_timestamp_seconds <= t <= (s.end_timestamp_seconds + 1e-4)
            ):
                matched_seg = s
                break

        if matched_seg is not None:
            activity_classes.append(matched_seg.activity_class)
            segment_ids.append(matched_seg.segment_id)
            sources.append(matched_seg.annotation_source)
        else:
            activity_classes.append(ACTIVITY_UNKNOWN)
            segment_ids.append(-1)
            sources.append("unannotated")

    df_copy["activity_class"] = activity_classes
    df_copy["segment_id"] = segment_ids
    df_copy["annotation_source"] = sources

    return df_copy


def calculate_activity_behaviour_distributions(
    mapped_df: pd.DataFrame,
    segments: Optional[List[TeachingActivitySegment]] = None,
    track_id: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate observed duration (seconds) and percentage share per behaviour for each activity.

    Computes durations strictly by merging consecutive identical behaviour predictions
    per track within each activity, eliminating double-counting of overlapping sequence windows.

    Args:
        mapped_df: DataFrame with predictions mapped to activities via map_predictions_to_activities.
        segments: Optional list of segments to ensure all annotated activities appear even with 0 observations.
        track_id: Optional track ID to filter for track-specific distribution.
            If None, aggregates across all tracks for classroom-level distribution.

    Returns:
        DataFrame with columns:
        [activity_class, behaviour_class, observed_duration_seconds, percentage_share,
         segment_count, sequence_count, track_id]
    """
    label_col = (
        "predicted_label"
        if "predicted_label" in mapped_df.columns
        else ("predicted_class" if "predicted_class" in mapped_df.columns else "dominant_behaviour")
    )

    # Determine activities to analyze
    activities_in_df = (
        [a for a in mapped_df["activity_class"].unique() if a != ACTIVITY_UNKNOWN]
        if not mapped_df.empty and "activity_class" in mapped_df.columns
        else []
    )
    activities_in_segs = [s.activity_class for s in (segments or [])]
    target_activities = []
    for a in TARGET_TEACHING_ACTIVITIES:
        if a in activities_in_df or a in activities_in_segs:
            target_activities.append(a)
    # If no activities detected yet, default to TARGET_TEACHING_ACTIVITIES
    if not target_activities:
        target_activities = list(TARGET_TEACHING_ACTIVITIES)

    # Filter by track if requested
    data_df = mapped_df.copy()
    if track_id is not None and not data_df.empty and "track_id" in data_df.columns:
        data_df = data_df[data_df["track_id"] == track_id]

    records: List[dict] = []

    for activity in target_activities:
        act_subset = data_df[data_df["activity_class"] == activity] if not data_df.empty else pd.DataFrame()

        # Track-wise segment extraction within this activity
        behaviour_durations: Dict[str, float] = {b: 0.0 for b in TARGET_BEHAVIOUR_CLASSES}
        behaviour_seg_counts: Dict[str, int] = {b: 0 for b in TARGET_BEHAVIOUR_CLASSES}
        behaviour_seq_counts: Dict[str, int] = {b: 0 for b in TARGET_BEHAVIOUR_CLASSES}

        if not act_subset.empty and "track_id" in act_subset.columns:
            tracks_in_act = act_subset["track_id"].unique()
            for tid in tracks_in_act:
                t_sub = act_subset[act_subset["track_id"] == tid].copy()
                if label_col in t_sub.columns:
                    for b in TARGET_BEHAVIOUR_CLASSES:
                        behaviour_seq_counts[b] += int((t_sub[label_col] == b).sum())

                # Merge contiguous predictions into segments for this track
                track_segments = merge_behaviour_segments(t_sub)
                for seg in track_segments:
                    b_cls = seg.behaviour_class
                    if b_cls in behaviour_durations:
                        behaviour_durations[b_cls] += seg.duration_seconds
                        behaviour_seg_counts[b_cls] += 1
                    else:
                        behaviour_durations[b_cls] = seg.duration_seconds
                        behaviour_seg_counts[b_cls] = 1

        total_activity_observed_dur = sum(behaviour_durations.values())

        for b_cls in TARGET_BEHAVIOUR_CLASSES:
            dur = behaviour_durations[b_cls]
            pct = (
                (dur / total_activity_observed_dur * 100.0)
                if total_activity_observed_dur > 0
                else 0.0
            )
            records.append(
                {
                    "activity_class": activity,
                    "behaviour_class": b_cls,
                    "observed_duration_seconds": round(dur, 2),
                    "percentage_share": round(pct, 2),
                    "segment_count": behaviour_seg_counts[b_cls],
                    "sequence_count": behaviour_seq_counts[b_cls],
                    "track_id": track_id if track_id is not None else "classroom",
                }
            )

    return pd.DataFrame(records)


def generate_activity_summary_table(
    distributions_df: pd.DataFrame,
    segments: List[TeachingActivitySegment],
    mapped_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create pedagogical summary table of instructional activities and observed behaviours.

    Includes explicit tie detection when multiple behaviours share the maximum duration.

    Args:
        distributions_df: DataFrame returned by calculate_activity_behaviour_distributions.
        segments: List of TeachingActivitySegment objects.
        mapped_df: Optional mapped predictions DataFrame for track count calculation.

    Returns:
        Summary DataFrame with columns:
        [Activity, Duration (s), Formatted Duration, Track Count, Total Observed (s),
         Dominant Behaviour, Dominant Share (%), Is Tie, Observation Count]
    """
    if distributions_df.empty:
        return pd.DataFrame(
            columns=[
                "Activity",
                "Duration (s)",
                "Formatted Duration",
                "Track Count",
                "Total Observed (s)",
                "Dominant Behaviour",
                "Dominant Share (%)",
                "Is Tie",
                "Observation Count",
            ]
        )

    # Activity durations from annotated segments
    activity_durations: Dict[str, float] = {}
    for s in segments:
        activity_durations[s.activity_class] = (
            activity_durations.get(s.activity_class, 0.0) + s.duration_seconds
        )

    # Track counts per activity
    activity_track_counts: Dict[str, int] = {}
    if mapped_df is not None and not mapped_df.empty and "activity_class" in mapped_df.columns:
        for act in mapped_df["activity_class"].unique():
            if act != ACTIVITY_UNKNOWN and "track_id" in mapped_df.columns:
                activity_track_counts[act] = int(
                    mapped_df[mapped_df["activity_class"] == act]["track_id"].nunique()
                )

    rows: List[dict] = []
    unique_activities = distributions_df["activity_class"].unique()

    for act in unique_activities:
        act_dist = distributions_df[distributions_df["activity_class"] == act]
        total_obs = float(act_dist["observed_duration_seconds"].sum())
        total_seq = int(act_dist["sequence_count"].sum())
        act_dur = activity_durations.get(act, 0.0)
        t_count = activity_track_counts.get(act, 0)

        # Dominant behaviour and tie detection
        max_dur = act_dist["observed_duration_seconds"].max()
        if max_dur > 0:
            top_behaviours = act_dist[
                act_dist["observed_duration_seconds"] >= (max_dur - 1e-3)
            ]["behaviour_class"].tolist()

            is_tie = bool(len(top_behaviours) > 1)
            if is_tie:
                dominant_str = " & ".join(top_behaviours) + " (Tied)"
            else:
                dominant_str = top_behaviours[0]

            dominant_pct = (
                round((max_dur / total_obs * 100.0), 1) if total_obs > 0 else 0.0
            )
        else:
            is_tie = False
            dominant_str = "None (No observations)"
            dominant_pct = 0.0

        rows.append(
            {
                "Activity": act,
                "Duration (s)": round(act_dur, 2),
                "Formatted Duration": format_timestamp_mmss(act_dur),
                "Track Count": t_count,
                "Total Observed (s)": round(total_obs, 2),
                "Dominant Behaviour": dominant_str,
                "Dominant Share (%)": f"{dominant_pct:.1f}%",
                "Is Tie": is_tie,
                "Observation Count": total_seq,
            }
        )

    return pd.DataFrame(rows)


def calculate_activity_transitions(
    mapped_df: pd.DataFrame,
    track_id: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate observable behaviour transition frequencies stratified by teaching activity.

    Examines where student behaviour shifted from one category to another within
    the boundary of each teaching activity.

    Args:
        mapped_df: Predictions DataFrame mapped to activities.
        track_id: Optional track ID to filter transitions for a specific track.

    Returns:
        DataFrame with columns:
        [activity_class, from_behaviour, to_behaviour, transition_label, count, track_id]
    """
    if mapped_df.empty or "activity_class" not in mapped_df.columns:
        return pd.DataFrame(
            columns=[
                "activity_class",
                "from_behaviour",
                "to_behaviour",
                "transition_label",
                "count",
                "track_id",
            ]
        )

    data_df = mapped_df.copy()
    if track_id is not None and "track_id" in data_df.columns:
        data_df = data_df[data_df["track_id"] == track_id]

    rows: List[dict] = []
    activities = [a for a in data_df["activity_class"].unique() if a != ACTIVITY_UNKNOWN]

    for act in activities:
        act_subset = data_df[data_df["activity_class"] == act]
        if act_subset.empty or "track_id" not in act_subset.columns:
            continue

        for tid in act_subset["track_id"].unique():
            t_sub = act_subset[act_subset["track_id"] == tid]
            transitions = calculate_behaviour_transitions(track_df=t_sub)

            for trans in transitions:
                rows.append(
                    {
                        "activity_class": act,
                        "from_behaviour": trans.from_behaviour,
                        "to_behaviour": trans.to_behaviour,
                        "transition_label": trans.label,
                        "count": trans.count,
                        "track_id": tid,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "activity_class",
                "from_behaviour",
                "to_behaviour",
                "transition_label",
                "count",
                "track_id",
            ]
        )

    res_df = pd.DataFrame(rows)
    if track_id is None:
        # Aggregate across tracks
        grouped = (
            res_df.groupby(["activity_class", "from_behaviour", "to_behaviour", "transition_label"])[
                "count"
            ]
            .sum()
            .reset_index()
        )
        grouped["track_id"] = "classroom"
        grouped.sort_values(by=["activity_class", "count"], ascending=[True, False], inplace=True)
        return grouped.reset_index(drop=True)

    res_df.sort_values(by=["activity_class", "count"], ascending=[True, False], inplace=True)
    return res_df.reset_index(drop=True)


def export_teaching_activity_results(
    video_id: str,
    model_type: str,
    segments: List[TeachingActivitySegment],
    summary_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    results_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Persist teaching activity analysis results to disk.

    Args:
        video_id: Video identifier stem.
        model_type: Temporal model used (e.g. 'lstm', 'rnn', 'gru').
        segments: TeachingActivitySegment list.
        summary_df: Summary table DataFrame.
        distribution_df: Behaviour distributions DataFrame.
        transitions_df: Transitions DataFrame.
        results_dir: Optional root results directory.

    Returns:
        Dict mapping artifact keys to their resolved file Paths.
    """
    out_dir = (results_dir or RESULTS_ACTIVITY_DIR) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    m_str = model_type.lower()

    summary_path = out_dir / f"activity_behaviour_summary_{m_str}.csv"
    dist_path = out_dir / f"activity_behaviour_distributions_{m_str}.csv"
    trans_path = out_dir / f"activity_transitions_{m_str}.csv"

    summary_df.to_csv(summary_path, index=False)
    distribution_df.to_csv(dist_path, index=False)
    transitions_df.to_csv(trans_path, index=False)

    return {
        "summary_csv": summary_path,
        "distribution_csv": dist_path,
        "transitions_csv": trans_path,
    }
