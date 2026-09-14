"""Observable Behaviour Trajectory Extraction and Analysis Module for EduPulse AI.

Provides:
1. Track-wise trajectory isolation and time-range filtering.
2. Consecutive identical prediction grouping into continuous BehaviourSegments.
3. Chronological Behaviour Transition tracking and frequency accounting.
4. Exact observed duration (seconds) and percentage of observed time calculation.
5. Missing observation and tracking gap detection.
6. Timestamp formatting (MM:SS) and comprehensive TrackTrajectorySummary aggregation.

Academic Scope:
All computations are strictly confined to observable classroom behaviour categories.
No inferences regarding internal mental, cognitive, or emotional states are made.
"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.behaviour.behaviour_labels import (
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
)


def format_timestamp_mmss(seconds: float) -> str:
    """Format numerical seconds as MM:SS string (e.g. 75.5 -> '01:15').
    
    Args:
        seconds: Non-negative elapsed timestamp in seconds.

    Returns:
        Formatted string 'MM:SS'.
    """
    if math.isnan(seconds) or seconds < 0:
        return "00:00"
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    rem_secs = total_seconds % 60
    return f"{minutes:02d}:{rem_secs:02d}"


@dataclass
class BehaviourSegment:
    """Continuous temporal segment of consecutive identical observable behaviour predictions."""

    segment_id: int
    track_id: int
    behaviour_class: str
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    start_frame_id: int
    end_frame_id: int
    start_sequence_id: int
    end_sequence_id: int
    mean_confidence: float
    sequence_count: int

    @property
    def start_time_formatted(self) -> str:
        return format_timestamp_mmss(self.start_timestamp_seconds)

    @property
    def end_time_formatted(self) -> str:
        return format_timestamp_mmss(self.end_timestamp_seconds)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_time_formatted"] = self.start_time_formatted
        d["end_time_formatted"] = self.end_time_formatted
        return d


@dataclass
class BehaviourTransition:
    """Observed transition between two consecutive behaviour categories."""

    from_behaviour: str
    to_behaviour: str
    count: int
    transition_timestamps: List[float]

    @property
    def label(self) -> str:
        return f"{self.from_behaviour} → {self.to_behaviour}"


@dataclass
class TrackingGap:
    """Temporal gap where tracked observations are discontinuous or missing."""

    gap_id: int
    track_id: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    start_sequence_id: int
    end_sequence_id: int

    @property
    def start_time_formatted(self) -> str:
        return format_timestamp_mmss(self.start_timestamp_seconds)

    @property
    def end_time_formatted(self) -> str:
        return format_timestamp_mmss(self.end_timestamp_seconds)


@dataclass
class TrackTrajectorySummary:
    """High-level summary metrics for an anonymous student track trajectory."""

    track_id: int
    model_type: str
    observation_start_seconds: float
    observation_end_seconds: float
    total_observed_duration_seconds: float
    categories_observed_count: int
    total_transitions_count: int
    total_segments_count: int
    gap_count: int
    total_gap_duration_seconds: float
    dominant_behaviour: str
    mean_confidence: float

    @property
    def observation_start_formatted(self) -> str:
        return format_timestamp_mmss(self.observation_start_seconds)

    @property
    def observation_end_formatted(self) -> str:
        return format_timestamp_mmss(self.observation_end_seconds)

    @property
    def start_time_formatted(self) -> str:
        return self.observation_start_formatted

    @property
    def end_time_formatted(self) -> str:
        return self.observation_end_formatted

    @property
    def total_observed_seconds(self) -> float:
        return self.total_observed_duration_seconds

    @property
    def distinct_behaviours_observed(self) -> int:
        return self.categories_observed_count

    @property
    def transition_count(self) -> int:
        return self.total_transitions_count

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observation_start_formatted"] = self.observation_start_formatted
        d["observation_end_formatted"] = self.observation_end_formatted
        d["start_time_formatted"] = self.start_time_formatted
        d["end_time_formatted"] = self.end_time_formatted
        return d


def extract_track_trajectory(
    predictions_df: pd.DataFrame,
    track_id: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    time_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Extract and chronologically filter predictions for a single anonymous track.

    Args:
        predictions_df: DataFrame of sequence predictions.
        track_id: Target anonymous student track ID.
        start_time: Optional minimum start timestamp in seconds.
        end_time: Optional maximum end timestamp in seconds.
        time_range: Optional (start_time, end_time) tuple for convenience.

    Returns:
        DataFrame sorted by start_timestamp_seconds for the specified track.
    """
    if time_range is not None:
        start_time = time_range[0]
        end_time = time_range[1]

    if predictions_df.empty or "track_id" not in predictions_df.columns:
        return pd.DataFrame()

    track_mask = predictions_df["track_id"] == track_id
    track_df = predictions_df[track_mask].copy()

    if track_df.empty:
        return track_df

    # Sort strictly chronologically
    sort_cols = [col for col in ["start_timestamp_seconds", "sequence_id"] if col in track_df.columns]
    if sort_cols:
        track_df = track_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Apply time range filtering
    if start_time is not None and "end_timestamp_seconds" in track_df.columns:
        track_df = track_df[track_df["end_timestamp_seconds"] >= start_time]

    if end_time is not None and "start_timestamp_seconds" in track_df.columns:
        track_df = track_df[track_df["start_timestamp_seconds"] <= end_time]

    return track_df.reset_index(drop=True)


def detect_tracking_gaps(
    track_df: pd.DataFrame,
    max_allowed_gap_seconds: float = 2.0,
) -> List[TrackingGap]:
    """Detect discontinuity gaps between consecutive observation windows for a track.

    Args:
        track_df: Chronologically sorted predictions DataFrame for a single track.
        max_allowed_gap_seconds: Threshold above which missing observations constitute a gap.

    Returns:
        List of TrackingGap objects.
    """
    gaps: List[TrackingGap] = []
    if len(track_df) < 2:
        return gaps

    for i in range(len(track_df) - 1):
        curr_row = track_df.iloc[i]
        next_row = track_df.iloc[i + 1]

        curr_end = float(curr_row.get("end_timestamp_seconds", 0.0))
        next_start = float(next_row.get("start_timestamp_seconds", 0.0))

        delta = next_start - curr_end
        if delta > max_allowed_gap_seconds:
            curr_seq = int(curr_row.get("sequence_id", i))
            next_seq = int(next_row.get("sequence_id", i + 1))
            tid = int(curr_row.get("track_id", 0))

            gaps.append(
                TrackingGap(
                    gap_id=len(gaps) + 1,
                    track_id=tid,
                    start_timestamp_seconds=round(curr_end, 3),
                    end_timestamp_seconds=round(next_start, 3),
                    duration_seconds=round(delta, 3),
                    start_sequence_id=curr_seq,
                    end_sequence_id=next_seq,
                )
            )

    return gaps


def merge_behaviour_segments(
    track_df: pd.DataFrame,
    max_gap_tolerance_seconds: float = 2.0,
) -> List[BehaviourSegment]:
    """Group consecutive identical behaviour predictions into continuous temporal segments.

    Where consecutive predictions share the same observable behaviour and no gap occurs,
    they are aggregated into a single BehaviourSegment spanning from the first window's
    start to the last window's end.

    Args:
        track_df: Chronologically sorted predictions DataFrame for a track.
        max_gap_tolerance_seconds: Tolerance for bridging contiguous windows.

    Returns:
        List of BehaviourSegment instances.
    """
    if track_df.empty:
        return []

    segments: List[BehaviourSegment] = []
    current_chunk: List[dict] = []

    label_col = "predicted_label" if "predicted_label" in track_df.columns else "dominant_behaviour"

    for _, row in track_df.iterrows():
        row_dict = row.to_dict()
        if not current_chunk:
            current_chunk.append(row_dict)
            continue

        prev_row = current_chunk[-1]
        prev_label = str(prev_row.get(label_col, CLASS_UNKNOWN))
        curr_label = str(row_dict.get(label_col, CLASS_UNKNOWN))

        prev_end = float(prev_row.get("end_timestamp_seconds", 0.0))
        curr_start = float(row_dict.get("start_timestamp_seconds", 0.0))
        gap = curr_start - prev_end

        # Merge if identical behaviour and contiguous (gap <= tolerance)
        if curr_label == prev_label and gap <= max_gap_tolerance_seconds:
            current_chunk.append(row_dict)
        else:
            # Finalize current segment
            seg = _create_segment_from_chunk(current_chunk, len(segments) + 1, label_col)
            segments.append(seg)
            current_chunk = [row_dict]

    if current_chunk:
        seg = _create_segment_from_chunk(current_chunk, len(segments) + 1, label_col)
        segments.append(seg)

    return segments


def _create_segment_from_chunk(
    chunk: List[dict],
    segment_id: int,
    label_col: str,
) -> BehaviourSegment:
    """Helper to convert a list of contiguous sequence records into a BehaviourSegment."""
    first = chunk[0]
    last = chunk[-1]

    tid = int(first.get("track_id", 0))
    b_class = str(first.get(label_col, CLASS_UNKNOWN))
    start_t = float(first.get("start_timestamp_seconds", 0.0))
    end_t = float(last.get("end_timestamp_seconds", 0.0))
    duration = max(0.0, end_t - start_t)

    start_f = int(first.get("start_frame_id", 0))
    end_f = int(last.get("end_frame_id", 0))

    start_s = int(first.get("sequence_id", 0))
    end_s = int(last.get("sequence_id", 0))

    confs = [float(r.get("confidence", 0.0)) for r in chunk if "confidence" in r]
    mean_conf = float(np.mean(confs)) if confs else 1.0

    return BehaviourSegment(
        segment_id=segment_id,
        track_id=tid,
        behaviour_class=b_class,
        start_timestamp_seconds=round(start_t, 3),
        end_timestamp_seconds=round(end_t, 3),
        duration_seconds=round(duration, 3),
        start_frame_id=start_f,
        end_frame_id=end_f,
        start_sequence_id=start_s,
        end_sequence_id=end_s,
        mean_confidence=round(mean_conf, 4),
        sequence_count=len(chunk),
    )


def calculate_behaviour_transitions(
    track_df_or_segments: Union[pd.DataFrame, List[BehaviourSegment], None] = None,
    track_df: Optional[pd.DataFrame] = None,
    segments: Optional[List[BehaviourSegment]] = None,
) -> List[BehaviourTransition]:
    """Calculate observable behaviour transition frequencies for a tracked individual.

    Identifies each point in time where the observed behaviour shifted from class A to class B.

    Args:
        track_df_or_segments: Optional DataFrame or list of BehaviourSegments.
        track_df: Optional predictions DataFrame.
        segments: Optional pre-computed list of BehaviourSegments (preferred).

    Returns:
        List of BehaviourTransition instances sorted by descending transition frequency.
    """
    if segments is None and isinstance(track_df_or_segments, list):
        segments = track_df_or_segments
    elif track_df is None and isinstance(track_df_or_segments, pd.DataFrame):
        track_df = track_df_or_segments

    if segments is None and track_df is not None:
        segments = merge_behaviour_segments(track_df)

    if not segments or len(segments) < 2:
        return []

    transition_dict: Dict[Tuple[str, str], List[float]] = {}

    for i in range(len(segments) - 1):
        from_b = segments[i].behaviour_class
        to_b = segments[i + 1].behaviour_class

        if from_b != to_b:
            pair = (from_b, to_b)
            t_time = segments[i + 1].start_timestamp_seconds
            if pair not in transition_dict:
                transition_dict[pair] = []
            transition_dict[pair].append(t_time)

    result: List[BehaviourTransition] = []
    for (from_b, to_b), timestamps in transition_dict.items():
        result.append(
            BehaviourTransition(
                from_behaviour=from_b,
                to_behaviour=to_b,
                count=len(timestamps),
                transition_timestamps=timestamps,
            )
        )

    # Sort descending by count, then alphabetically
    result.sort(key=lambda t: (-t.count, t.from_behaviour, t.to_behaviour))
    return result


def calculate_behaviour_durations_and_distribution(
    segments: List[BehaviourSegment],
    total_observed_time: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Calculate total observed duration and percentage of observed time per category.

    Args:
        segments: List of BehaviourSegment instances.
        total_observed_time: Optional denominator for percentage calculation.
            If None, sums the duration across all non-gap segments.

    Returns:
        Tuple of:
        - summary_df: DataFrame with columns:
            [Observable Behaviour, Observed Duration (sec), Percentage of Observed Time, Segment Count]
        - durations_dict: {class_name: duration_in_seconds}
        - percentages_dict: {class_name: percentage_0_to_100}
    """
    durations: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for seg in segments:
        cls_name = seg.behaviour_class
        durations[cls_name] = durations.get(cls_name, 0.0) + seg.duration_seconds
        counts[cls_name] = counts.get(cls_name, 0) + 1

    total_time = total_observed_time or sum(durations.values())
    percentages: Dict[str, float] = {}

    rows = []
    for cls_name, dur in sorted(durations.items(), key=lambda x: -x[1]):
        pct = (dur / total_time * 100.0) if total_time > 0 else 0.0
        percentages[cls_name] = round(pct, 2)
        rows.append(
            {
                "Observable Behaviour": cls_name,
                "Observed Duration (sec)": round(dur, 2),
                "Percentage of Observed Time": f"{pct:.1f}%",
                "Segment Count": counts.get(cls_name, 0),
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df, durations, percentages


def generate_track_summary(
    track_id: int,
    model_type: str,
    track_df: pd.DataFrame,
    segments: List[BehaviourSegment],
    transitions: List[BehaviourTransition],
    gaps: List[TrackingGap],
) -> TrackTrajectorySummary:
    """Aggregate high-level spatial-temporal trajectory statistics for an anonymous track.

    Args:
        track_id: Track ID integer.
        model_type: Name of model used (e.g. 'LSTM').
        track_df: Predictions DataFrame for track.
        segments: Merged BehaviourSegments.
        transitions: Computed BehaviourTransitions.
        gaps: Detected TrackingGaps.

    Returns:
        TrackTrajectorySummary dataclass instance.
    """
    if track_df.empty:
        return TrackTrajectorySummary(
            track_id=track_id,
            model_type=model_type.upper(),
            observation_start_seconds=0.0,
            observation_end_seconds=0.0,
            total_observed_duration_seconds=0.0,
            categories_observed_count=0,
            total_transitions_count=0,
            total_segments_count=0,
            gap_count=0,
            total_gap_duration_seconds=0.0,
            dominant_behaviour=CLASS_UNKNOWN,
            mean_confidence=0.0,
        )

    start_t = float(track_df["start_timestamp_seconds"].min())
    end_t = float(track_df["end_timestamp_seconds"].max())

    total_obs_dur = sum(s.duration_seconds for s in segments)
    distinct_classes = set(s.behaviour_class for s in segments)

    total_transitions = sum(t.count for t in transitions)
    total_gap_dur = sum(g.duration_seconds for g in gaps)

    # Dominant behaviour: category with highest total duration
    durations: Dict[str, float] = {}
    for s in segments:
        durations[s.behaviour_class] = durations.get(s.behaviour_class, 0.0) + s.duration_seconds
    dominant = max(durations.keys(), key=lambda k: durations[k]) if durations else CLASS_UNKNOWN

    confs = track_df["confidence"].dropna() if "confidence" in track_df.columns else []
    mean_conf = float(confs.mean()) if len(confs) > 0 else 1.0

    return TrackTrajectorySummary(
        track_id=track_id,
        model_type=model_type.upper(),
        observation_start_seconds=round(start_t, 3),
        observation_end_seconds=round(end_t, 3),
        total_observed_duration_seconds=round(total_obs_dur, 3),
        categories_observed_count=len(distinct_classes),
        total_transitions_count=total_transitions,
        total_segments_count=len(segments),
        gap_count=len(gaps),
        total_gap_duration_seconds=round(total_gap_dur, 3),
        dominant_behaviour=dominant,
        mean_confidence=round(mean_conf, 4),
    )
