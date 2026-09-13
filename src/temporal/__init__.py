"""Temporal sequence creation package for EduPulse AI.

Provides sequence generation, sliding-window chunking, PyTorch Dataset adapter,
and timeline visualization for temporal engagement analysis.
"""

from src.temporal.sequence_generator import (
    ClassroomSequenceDataset,
    SequenceSummary,
    TemporalSequenceGenerator,
    run_temporal_sequence_creation,
    sequences_to_tensor,
)
from src.temporal.visualization import (
    create_sequence_timeline_figure,
    create_track_coverage_figure,
)

__all__ = [
    "TemporalSequenceGenerator",
    "SequenceSummary",
    "run_temporal_sequence_creation",
    "sequences_to_tensor",
    "ClassroomSequenceDataset",
    "create_sequence_timeline_figure",
    "create_track_coverage_figure",
]
