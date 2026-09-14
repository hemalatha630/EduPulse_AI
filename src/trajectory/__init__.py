"""Observable Behaviour Trajectory Package for EduPulse AI (Feature 9).

Provides:
- Trajectory extraction, chronological track isolation, and time-range filtering
- Behaviour segment merging across contiguous identical predictions
- Chronological behaviour transition counting
- Observed duration (seconds) and percentage of observed time calculation
- Tracking gap detection
- Model inference reuse without retraining (load_or_generate_trajectory_predictions)
- Categorical Gantt-style timeline visualization
- Side-by-side model comparison timelines (RNN vs LSTM vs GRU)
- Classroom multi-track overview timelines
- Structured CSV trajectory export
"""

from src.trajectory.extractor import (
    BehaviourSegment,
    BehaviourTransition,
    TrackTrajectorySummary,
    TrackingGap,
    calculate_behaviour_durations_and_distribution,
    calculate_behaviour_transitions,
    detect_tracking_gaps,
    extract_track_trajectory,
    format_timestamp_mmss,
    generate_track_summary,
    merge_behaviour_segments,
)
from src.trajectory.manager import (
    export_behaviour_trajectories_csv,
    load_or_generate_trajectory_predictions,
)
from src.trajectory.visualization import (
    create_categorical_timeline_figure,
    create_classroom_overview_figure,
    create_duration_distribution_figure,
    create_model_comparison_timeline_figure,
    create_transitions_figure,
    get_behaviour_hex_color,
)

__all__ = [
    # Extractor & Dataclasses
    "BehaviourSegment",
    "BehaviourTransition",
    "TrackingGap",
    "TrackTrajectorySummary",
    "format_timestamp_mmss",
    "extract_track_trajectory",
    "detect_tracking_gaps",
    "merge_behaviour_segments",
    "calculate_behaviour_transitions",
    "calculate_behaviour_durations_and_distribution",
    "generate_track_summary",
    # Manager & Persistence
    "load_or_generate_trajectory_predictions",
    "export_behaviour_trajectories_csv",
    # Visualizations
    "create_categorical_timeline_figure",
    "create_duration_distribution_figure",
    "create_transitions_figure",
    "create_model_comparison_timeline_figure",
    "create_classroom_overview_figure",
    "get_behaviour_hex_color",
]
