"""Teaching Activity Analysis Module (Feature 10) for EduPulse AI.

Provides comprehensive domain logic, validation, temporal mapping, distribution accounting,
and publication-quality visualization for classroom instructional activity analysis.
"""

from src.activity.activity_labels import (
    ACTIVITY_DESCRIPTIONS,
    ACTIVITY_DISCUSSION,
    ACTIVITY_LECTURE,
    ACTIVITY_PRESENTATION,
    ACTIVITY_PROBLEM_SOLVING,
    ACTIVITY_SOURCE_MANUAL,
    ACTIVITY_SOURCE_METADATA,
    ACTIVITY_UNKNOWN,
    TARGET_TEACHING_ACTIVITIES,
    VALID_ANNOTATION_SOURCES,
    get_activity_description,
    get_activity_hex,
    get_activity_rgb,
)
from src.activity.analyzer import (
    calculate_activity_behaviour_distributions,
    calculate_activity_transitions,
    export_teaching_activity_results,
    generate_activity_summary_table,
    map_predictions_to_activities,
)
from src.activity.manager import (
    TeachingActivitySegment,
    create_default_demo_segments,
    load_teaching_activity_segments,
    save_teaching_activity_segments,
    validate_activity_segments,
)
from src.activity.visualization import (
    create_activity_behaviour_distribution_figure,
    create_activity_behaviour_heatmap_figure,
    create_activity_timeline_figure,
    create_track_activity_figure,
    get_behaviour_hex_color,
)

__all__ = [
    # Activity Labels & Metadata
    "ACTIVITY_LECTURE",
    "ACTIVITY_DISCUSSION",
    "ACTIVITY_PROBLEM_SOLVING",
    "ACTIVITY_PRESENTATION",
    "TARGET_TEACHING_ACTIVITIES",
    "ACTIVITY_DESCRIPTIONS",
    "ACTIVITY_SOURCE_MANUAL",
    "ACTIVITY_SOURCE_METADATA",
    "VALID_ANNOTATION_SOURCES",
    "ACTIVITY_UNKNOWN",
    "get_activity_hex",
    "get_activity_rgb",
    "get_activity_description",
    # Manager & Segments
    "TeachingActivitySegment",
    "validate_activity_segments",
    "load_teaching_activity_segments",
    "save_teaching_activity_segments",
    "create_default_demo_segments",
    # Analyzer & Distributions
    "map_predictions_to_activities",
    "calculate_activity_behaviour_distributions",
    "generate_activity_summary_table",
    "calculate_activity_transitions",
    "export_teaching_activity_results",
    # Visualizations
    "create_activity_timeline_figure",
    "create_activity_behaviour_distribution_figure",
    "create_activity_behaviour_heatmap_figure",
    "create_track_activity_figure",
    "get_behaviour_hex_color",
]
