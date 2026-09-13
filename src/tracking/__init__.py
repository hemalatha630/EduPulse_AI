"""Tracking package interface for EduPulse AI.

Provides multi-object tracking (ByteTrack / BoT-SORT) for classroom video analysis.
"""

from src.tracking.tracker import (
    PersonTracker,
    TrackResult,
    TrackingSummary,
    get_track_color,
    run_tracking_on_frames,
)

__all__ = [
    "PersonTracker",
    "TrackResult",
    "TrackingSummary",
    "get_track_color",
    "run_tracking_on_frames",
]
