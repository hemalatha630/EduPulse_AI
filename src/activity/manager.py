"""Teaching Activity Segment Annotation Management and Validation Module.

Provides:
1. TeachingActivitySegment dataclass with formatted timestamp accessors.
2. Robust validation rules: start < end, non-negative, non-overlapping, bounds checking.
3. Loading and saving segment annotations from/to disk (data/processed/<video_id>/teaching_activity_segments.csv).
4. Template generation for standard demonstration videos.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.activity.activity_labels import (
    ACTIVITY_DISCUSSION,
    ACTIVITY_LECTURE,
    ACTIVITY_PRESENTATION,
    ACTIVITY_PROBLEM_SOLVING,
    ACTIVITY_SOURCE_MANUAL,
    TARGET_TEACHING_ACTIVITIES,
)
from src.config import (
    PROCESSED_DIR,
    TEACHING_ACTIVITY_SEGMENTS_FILENAME,
)
from src.trajectory.extractor import format_timestamp_mmss


@dataclass
class TeachingActivitySegment:
    """Designated time interval annotated with an instructional classroom teaching activity."""

    segment_id: int
    video_id: str
    activity_class: str
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    annotation_source: str = ACTIVITY_SOURCE_MANUAL

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


def validate_activity_segments(
    segments: List[TeachingActivitySegment],
    video_duration_seconds: Optional[float] = None,
) -> None:
    """Validate list of teaching activity segments according to strict temporal rules.

    Rules:
    1. start_timestamp_seconds < end_timestamp_seconds
    2. start_timestamp_seconds >= 0.0
    3. If video_duration_seconds is provided, end_timestamp_seconds <= video_duration_seconds
    4. activity_class must be in TARGET_TEACHING_ACTIVITIES
    5. No overlapping segments permitted.

    Args:
        segments: List of TeachingActivitySegment objects.
        video_duration_seconds: Optional max allowed duration of video.

    Raises:
        ValueError: If any validation condition is violated.
    """
    if not segments:
        return

    # Check individual segment integrity
    for s in segments:
        if math.isnan(s.start_timestamp_seconds) or math.isnan(s.end_timestamp_seconds):
            raise ValueError(f"Segment #{s.segment_id} contains invalid NaN timestamps.")

        if s.start_timestamp_seconds < 0.0:
            raise ValueError(
                f"Segment #{s.segment_id} has negative start time ({s.start_timestamp_seconds:.2f}s). "
                "Timestamps must be non-negative."
            )

        if s.start_timestamp_seconds >= s.end_timestamp_seconds:
            raise ValueError(
                f"Segment #{s.segment_id} ({s.activity_class}) has start time ({s.start_timestamp_seconds:.2f}s) "
                f">= end time ({s.end_timestamp_seconds:.2f}s). Start time must precede end time."
            )

        if video_duration_seconds is not None and video_duration_seconds > 0:
            # Allow tiny 0.1s float epsilon tolerance for video boundary
            if s.end_timestamp_seconds > (video_duration_seconds + 0.1):
                raise ValueError(
                    f"Segment #{s.segment_id} end time ({s.end_timestamp_seconds:.2f}s) exceeds "
                    f"video duration ({video_duration_seconds:.2f}s)."
                )

        if s.activity_class not in TARGET_TEACHING_ACTIVITIES:
            raise ValueError(
                f"Segment #{s.segment_id} has invalid activity '{s.activity_class}'. "
                f"Allowed activities: {TARGET_TEACHING_ACTIVITIES}."
            )

    # Check for overlapping segments
    sorted_segs = sorted(segments, key=lambda x: x.start_timestamp_seconds)
    for i in range(len(sorted_segs) - 1):
        curr_s = sorted_segs[i]
        next_s = sorted_segs[i + 1]

        # Overlap occurs if current segment ends after next segment starts
        if curr_s.end_timestamp_seconds > next_s.start_timestamp_seconds + 1e-4:
            raise ValueError(
                f"Overlapping activity segments detected: "
                f"Segment #{curr_s.segment_id} ({curr_s.activity_class} [{curr_s.start_time_formatted} - {curr_s.end_time_formatted}]) "
                f"overlaps with Segment #{next_s.segment_id} ({next_s.activity_class} [{next_s.start_time_formatted} - {next_s.end_time_formatted}]). "
                "Teaching activities must not overlap in time."
            )


def load_teaching_activity_segments(
    video_id: str,
    processed_dir: Optional[Path] = None,
) -> List[TeachingActivitySegment]:
    """Load teaching activity segment annotations for a video from disk.

    Args:
        video_id: Video identifier stem.
        processed_dir: Root directory for processed artifacts.

    Returns:
        List of TeachingActivitySegment objects, sorted chronologically.
    """
    p_dir = (processed_dir or PROCESSED_DIR) / video_id
    csv_path = p_dir / TEACHING_ACTIVITY_SEGMENTS_FILENAME

    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        if df.empty or "activity_class" not in df.columns:
            return []

        segments = []
        for idx, row in df.iterrows():
            start_t = float(row.get("start_timestamp_seconds", row.get("start_timestamp", 0.0)))
            end_t = float(row.get("end_timestamp_seconds", row.get("end_timestamp", 0.0)))
            dur_t = float(row.get("duration_seconds", end_t - start_t))
            segments.append(
                TeachingActivitySegment(
                    segment_id=int(row.get("segment_id", idx + 1)),
                    video_id=str(row.get("video_id", video_id)),
                    activity_class=str(row.get("activity_class", ACTIVITY_LECTURE)),
                    start_timestamp_seconds=round(start_t, 3),
                    end_timestamp_seconds=round(end_t, 3),
                    duration_seconds=round(dur_t, 3),
                    annotation_source=str(row.get("annotation_source", ACTIVITY_SOURCE_MANUAL)),
                )
            )

        segments.sort(key=lambda s: s.start_timestamp_seconds)
        return segments
    except Exception:
        return []


def save_teaching_activity_segments(
    video_id: str,
    segments: List[TeachingActivitySegment],
    processed_dir: Optional[Path] = None,
    video_duration_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """Validate and persist teaching activity segment annotations to disk.

    Args:
        video_id: Video identifier stem.
        segments: List of TeachingActivitySegment objects to persist.
        processed_dir: Root directory for processed artifacts.
        video_duration_seconds: Optional duration of video for boundary verification.

    Returns:
        Saved DataFrame of activity segments.
    """
    validate_activity_segments(segments, video_duration_seconds=video_duration_seconds)

    p_dir = (processed_dir or PROCESSED_DIR) / video_id
    p_dir.mkdir(parents=True, exist_ok=True)
    csv_path = p_dir / TEACHING_ACTIVITY_SEGMENTS_FILENAME

    if not segments:
        empty_df = pd.DataFrame(
            columns=[
                "video_id",
                "segment_id",
                "activity_class",
                "start_timestamp_seconds",
                "end_timestamp_seconds",
                "duration_seconds",
                "annotation_source",
            ]
        )
        empty_df.to_csv(csv_path, index=False)
        return empty_df

    sorted_segs = sorted(segments, key=lambda s: s.start_timestamp_seconds)
    records = []
    for idx, s in enumerate(sorted_segs, 1):
        s.segment_id = idx
        records.append(
            {
                "video_id": video_id,
                "segment_id": idx,
                "activity_class": s.activity_class,
                "start_timestamp_seconds": round(s.start_timestamp_seconds, 3),
                "end_timestamp_seconds": round(s.end_timestamp_seconds, 3),
                "duration_seconds": round(s.end_timestamp_seconds - s.start_timestamp_seconds, 3),
                "annotation_source": s.annotation_source,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df


def create_default_demo_segments(
    video_id: str,
    video_duration_seconds: float = 3.0,
) -> List[TeachingActivitySegment]:
    """Generate balanced, valid demo teaching activity segments across video duration.

    Strictly labelled with annotation_source = 'manual' to indicate manual template source.

    Args:
        video_id: Video identifier stem.
        video_duration_seconds: Duration of the demo video in seconds.

    Returns:
        List of TeachingActivitySegments partitioning the video duration.
    """
    dur = max(1.0, float(video_duration_seconds))

    # Divide video into two or three instructional intervals
    if dur >= 4.0:
        t1 = round(dur * 0.4, 2)
        t2 = round(dur * 0.75, 2)
        return [
            TeachingActivitySegment(
                segment_id=1,
                video_id=video_id,
                activity_class=ACTIVITY_LECTURE,
                start_timestamp_seconds=0.0,
                end_timestamp_seconds=t1,
                duration_seconds=t1,
                annotation_source=ACTIVITY_SOURCE_MANUAL,
            ),
            TeachingActivitySegment(
                segment_id=2,
                video_id=video_id,
                activity_class=ACTIVITY_DISCUSSION,
                start_timestamp_seconds=t1,
                end_timestamp_seconds=t2,
                duration_seconds=round(t2 - t1, 2),
                annotation_source=ACTIVITY_SOURCE_MANUAL,
            ),
            TeachingActivitySegment(
                segment_id=3,
                video_id=video_id,
                activity_class=ACTIVITY_PROBLEM_SOLVING,
                start_timestamp_seconds=t2,
                end_timestamp_seconds=dur,
                duration_seconds=round(dur - t2, 2),
                annotation_source=ACTIVITY_SOURCE_MANUAL,
            ),
        ]
    else:
        # Standard 2-interval split for short demo recordings
        mid = round(dur / 2.0, 2)
        return [
            TeachingActivitySegment(
                segment_id=1,
                video_id=video_id,
                activity_class=ACTIVITY_LECTURE,
                start_timestamp_seconds=0.0,
                end_timestamp_seconds=mid,
                duration_seconds=mid,
                annotation_source=ACTIVITY_SOURCE_MANUAL,
            ),
            TeachingActivitySegment(
                segment_id=2,
                video_id=video_id,
                activity_class=ACTIVITY_DISCUSSION,
                start_timestamp_seconds=mid,
                end_timestamp_seconds=dur,
                duration_seconds=round(dur - mid, 2),
                annotation_source=ACTIVITY_SOURCE_MANUAL,
            ),
        ]
