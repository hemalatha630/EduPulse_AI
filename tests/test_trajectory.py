"""Comprehensive Unit and Integration Tests for Feature 9: Observable Behaviour Trajectory.

Tests:
1. Timestamp formatting (MM:SS).
2. Track-wise trajectory isolation & time filtering.
3. Tracking gap detection.
4. Consecutive behaviour segment merging.
5. Chronological transition calculation.
6. Observed duration and percentage of observed time calculation.
7. Track trajectory summary generation.
8. Unknown / Uncertain label preservation.
9. Zero-retraining prediction reuse across real classroom data.
10. Trajectory CSV export schema & data integrity.
11. Visualizations: categorical timeline, duration distribution, transitions, model comparison, classroom overview.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behaviour.behaviour_labels import (
    CLASS_HEAD_DOWN,
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_READING_WRITING,
    CLASS_UNKNOWN,
)
from src.config import (
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
    MODELS_TEMPORAL_DIR,
    PROCESSED_DIR,
)
from src.trajectory.extractor import (
    BehaviourSegment,
    BehaviourTransition,
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
)


@pytest.fixture
def sample_predictions_df() -> pd.DataFrame:
    """Fixture returning sample sequence-level predictions across multiple tracks."""
    records = [
        # Track 1: Reading (0.0 - 1.5), Reading (0.5 - 2.0), Looking (2.0 - 3.5), Looking (2.5 - 4.0)
        {"model": "LSTM", "video_id": "test_vid", "track_id": 1, "sequence_id": 0, "start_frame_id": 1, "end_frame_id": 10, "start_timestamp_seconds": 0.0, "end_timestamp_seconds": 1.5, "duration_seconds": 1.5, "predicted_label": CLASS_READING_WRITING, "confidence": 0.90},
        {"model": "LSTM", "video_id": "test_vid", "track_id": 1, "sequence_id": 1, "start_frame_id": 3, "end_frame_id": 12, "start_timestamp_seconds": 0.5, "end_timestamp_seconds": 2.0, "duration_seconds": 1.5, "predicted_label": CLASS_READING_WRITING, "confidence": 0.85},
        {"model": "LSTM", "video_id": "test_vid", "track_id": 1, "sequence_id": 2, "start_frame_id": 5, "end_frame_id": 14, "start_timestamp_seconds": 2.0, "end_timestamp_seconds": 3.5, "duration_seconds": 1.5, "predicted_label": CLASS_LOOKING_TOWARD_INSTRUCTION, "confidence": 0.88},
        {"model": "LSTM", "video_id": "test_vid", "track_id": 1, "sequence_id": 3, "start_frame_id": 7, "end_frame_id": 16, "start_timestamp_seconds": 2.5, "end_timestamp_seconds": 4.0, "duration_seconds": 1.5, "predicted_label": CLASS_LOOKING_TOWARD_INSTRUCTION, "confidence": 0.92},
        # Track 2: Interacting (0.0 - 1.5), Tracking gap, Looking Away (6.0 - 7.5)
        {"model": "LSTM", "video_id": "test_vid", "track_id": 2, "sequence_id": 4, "start_frame_id": 1, "end_frame_id": 10, "start_timestamp_seconds": 0.0, "end_timestamp_seconds": 1.5, "duration_seconds": 1.5, "predicted_label": CLASS_INTERACTING_WITH_PEERS, "confidence": 0.75},
        {"model": "LSTM", "video_id": "test_vid", "track_id": 2, "sequence_id": 5, "start_frame_id": 20, "end_frame_id": 30, "start_timestamp_seconds": 6.0, "end_timestamp_seconds": 7.5, "duration_seconds": 1.5, "predicted_label": CLASS_LOOKING_AWAY, "confidence": 0.80},
    ]
    return pd.DataFrame(records)


class TestTrajectoryExtractor:
    """Unit tests for trajectory extraction, segmenting, transitions, and duration."""

    def test_format_timestamp_mmss(self):
        """Verify seconds to MM:SS formatting."""
        assert format_timestamp_mmss(0.0) == "00:00"
        assert format_timestamp_mmss(59.4) == "00:59"
        assert format_timestamp_mmss(60.0) == "01:00"
        assert format_timestamp_mmss(75.5) == "01:15"
        assert format_timestamp_mmss(3605.0) == "60:05"
        assert format_timestamp_mmss(-5.0) == "00:00"
        assert format_timestamp_mmss(float("nan")) == "00:00"

    def test_extract_track_trajectory_isolation(self, sample_predictions_df):
        """Verify trajectory extraction isolates only the requested track."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        assert len(t1_df) == 4
        assert (t1_df["track_id"] == 1).all()
        # Verify chronological ordering
        assert t1_df["start_timestamp_seconds"].is_monotonic_increasing

        t2_df = extract_track_trajectory(sample_predictions_df, track_id=2)
        assert len(t2_df) == 2
        assert (t2_df["track_id"] == 2).all()

        # Non-existent track
        t99_df = extract_track_trajectory(sample_predictions_df, track_id=99)
        assert t99_df.empty

    def test_extract_track_trajectory_time_range_filter(self, sample_predictions_df):
        """Verify time-range filtering clips to the specified interval."""
        # Track 1 has sequences from 0.0 to 4.0
        filtered_df = extract_track_trajectory(sample_predictions_df, track_id=1, start_time=1.8, end_time=3.0)
        assert len(filtered_df) > 0
        assert (filtered_df["end_timestamp_seconds"] >= 1.8).all()
        assert (filtered_df["start_timestamp_seconds"] <= 3.0).all()

    def test_detect_tracking_gaps(self, sample_predictions_df):
        """Verify tracking gap detection when interval exceeds threshold."""
        # Track 1 has contiguous sequences (no gap > 2.0s)
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        t1_gaps = detect_tracking_gaps(t1_df, max_allowed_gap_seconds=2.0)
        assert len(t1_gaps) == 0

        # Track 2 has sequences: 0.0-1.5 and 6.0-7.5 (delta = 4.5s > 2.0s)
        t2_df = extract_track_trajectory(sample_predictions_df, track_id=2)
        t2_gaps = detect_tracking_gaps(t2_df, max_allowed_gap_seconds=2.0)
        assert len(t2_gaps) == 1
        gap = t2_gaps[0]
        assert gap.track_id == 2
        assert gap.start_timestamp_seconds == 1.5
        assert gap.end_timestamp_seconds == 6.0
        assert gap.duration_seconds == 4.5
        assert gap.start_time_formatted == "00:01"
        assert gap.end_time_formatted == "00:06"

    def test_merge_behaviour_segments(self, sample_predictions_df):
        """Verify consecutive identical predictions are grouped into discrete segments."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        segments = merge_behaviour_segments(t1_df)

        # Track 1 has 2 Reading windows, then 2 Looking windows -> exactly 2 segments
        assert len(segments) == 2

        seg1 = segments[0]
        assert seg1.segment_id == 1
        assert seg1.track_id == 1
        assert seg1.behaviour_class == CLASS_READING_WRITING
        assert seg1.start_timestamp_seconds == 0.0
        assert seg1.end_timestamp_seconds == 2.0
        assert seg1.duration_seconds == 2.0
        assert seg1.sequence_count == 2
        assert seg1.mean_confidence == pytest.approx(0.875, abs=1e-3)

        seg2 = segments[1]
        assert seg2.segment_id == 2
        assert seg2.track_id == 1
        assert seg2.behaviour_class == CLASS_LOOKING_TOWARD_INSTRUCTION
        assert seg2.start_timestamp_seconds == 2.0
        assert seg2.end_timestamp_seconds == 4.0
        assert seg2.duration_seconds == 2.0
        assert seg2.sequence_count == 2
        assert seg2.mean_confidence == pytest.approx(0.90, abs=1e-3)

    def test_calculate_behaviour_transitions(self, sample_predictions_df):
        """Verify chronological transitions between consecutive observable behaviours."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        transitions = calculate_behaviour_transitions(t1_df)

        # Track 1: Reading -> Looking (1 transition)
        assert len(transitions) == 1
        trans = transitions[0]
        assert trans.from_behaviour == CLASS_READING_WRITING
        assert trans.to_behaviour == CLASS_LOOKING_TOWARD_INSTRUCTION
        assert trans.count == 1
        assert trans.transition_timestamps == [2.0]
        assert trans.label == f"{CLASS_READING_WRITING} → {CLASS_LOOKING_TOWARD_INSTRUCTION}"

    def test_calculate_durations_and_distribution(self, sample_predictions_df):
        """Verify observed duration and percentage of time calculations."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        segments = merge_behaviour_segments(t1_df)

        summary_df, durations, percentages = calculate_behaviour_durations_and_distribution(segments)

        assert len(summary_df) == 2
        assert durations[CLASS_READING_WRITING] == 2.0
        assert durations[CLASS_LOOKING_TOWARD_INSTRUCTION] == 2.0

        # Percentages must sum to 100%
        assert sum(percentages.values()) == pytest.approx(100.0, abs=0.1)
        assert percentages[CLASS_READING_WRITING] == 50.0
        assert percentages[CLASS_LOOKING_TOWARD_INSTRUCTION] == 50.0

        # Check DataFrame schema
        expected_cols = [
            "Observable Behaviour",
            "Observed Duration (sec)",
            "Percentage of Observed Time",
            "Segment Count",
        ]
        assert list(summary_df.columns) == expected_cols

    def test_generate_track_summary(self, sample_predictions_df):
        """Verify high-level track trajectory summary aggregation."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        segments = merge_behaviour_segments(t1_df)
        transitions = calculate_behaviour_transitions(segments=segments)
        gaps = detect_tracking_gaps(t1_df)

        summary = generate_track_summary(
            track_id=1,
            model_type="LSTM",
            track_df=t1_df,
            segments=segments,
            transitions=transitions,
            gaps=gaps,
        )

        assert summary.track_id == 1
        assert summary.model_type == "LSTM"
        assert summary.observation_start_seconds == 0.0
        assert summary.observation_end_seconds == 4.0
        assert summary.total_observed_duration_seconds == 4.0
        assert summary.categories_observed_count == 2
        assert summary.total_transitions_count == 1
        assert summary.total_segments_count == 2
        assert summary.gap_count == 0
        assert summary.observation_start_formatted == "00:00"
        assert summary.observation_end_formatted == "00:04"

    def test_unknown_uncertain_label_preservation(self):
        """Verify Unknown/Uncertain predictions are preserved and not forced into canonical 6."""
        df = pd.DataFrame([
            {"track_id": 3, "sequence_id": 0, "start_timestamp_seconds": 0.0, "end_timestamp_seconds": 1.5, "predicted_label": CLASS_READING_WRITING, "confidence": 0.9},
            {"track_id": 3, "sequence_id": 1, "start_timestamp_seconds": 1.5, "end_timestamp_seconds": 3.0, "predicted_label": CLASS_UNKNOWN, "confidence": 0.4},
            {"track_id": 3, "sequence_id": 2, "start_timestamp_seconds": 3.0, "end_timestamp_seconds": 4.5, "predicted_label": CLASS_LOOKING_AWAY, "confidence": 0.8},
        ])
        segments = merge_behaviour_segments(df)
        assert len(segments) == 3
        assert segments[1].behaviour_class == CLASS_UNKNOWN


class TestTrajectoryManagerAndExport:
    """Integration tests for prediction loading, zero-retraining inference, and CSV export."""

    def test_load_or_generate_predictions_real_data(self):
        """Verify loading/generating predictions on actual classroom demo data."""
        demo_id = "classroom_lecture_demo"
        meta_path = PROCESSED_DIR / demo_id / "temporal_sequences_metadata.csv"
        ckpt_path = MODELS_TEMPORAL_DIR / "lstm_best.pt"

        if not (meta_path.exists() and ckpt_path.exists()):
            pytest.skip("classroom_lecture_demo data and LSTM checkpoint required")

        preds_df = load_or_generate_trajectory_predictions(
            video_id=demo_id,
            model_type=MODEL_LSTM,
        )

        assert not preds_df.empty
        assert len(preds_df) == 50  # 50 sequences generated in Feature 7
        assert "track_id" in preds_df.columns
        assert "predicted_label" in preds_df.columns
        assert "confidence" in preds_df.columns
        # Verify multiple tracks exist in predictions
        unique_tracks = preds_df["track_id"].unique()
        assert len(unique_tracks) > 5

    def test_export_behaviour_trajectories_csv(self, tmp_path, sample_predictions_df):
        """Verify enriched CSV export with sequence and segment metadata."""
        out_csv = tmp_path / "behaviour_trajectories.csv"
        exported_df = export_behaviour_trajectories_csv(
            video_id="test_vid",
            predictions_df=sample_predictions_df,
            output_csv_path=out_csv,
        )

        assert out_csv.exists()
        assert len(exported_df) == len(sample_predictions_df)
        assert "segment_id" in exported_df.columns
        assert "segment_start_timestamp_seconds" in exported_df.columns
        assert "segment_end_timestamp_seconds" in exported_df.columns
        assert "segment_duration_seconds" in exported_df.columns


class TestTrajectoryVisualizations:
    """Unit tests verifying all Feature 9 Matplotlib figures render properly."""

    def test_create_categorical_timeline_figure(self, sample_predictions_df):
        """Verify categorical timeline figure generation."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        segments = merge_behaviour_segments(t1_df)
        gaps = detect_tracking_gaps(t1_df)

        fig = create_categorical_timeline_figure(segments, gaps=gaps, track_id=1, model_name="LSTM")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_duration_distribution_figure(self):
        """Verify observed duration bar chart figure generation."""
        durations = {CLASS_READING_WRITING: 25.0, CLASS_LOOKING_AWAY: 10.0}
        percentages = {CLASS_READING_WRITING: 71.4, CLASS_LOOKING_AWAY: 28.6}

        fig = create_duration_distribution_figure(durations, percentages, track_id=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_transitions_figure(self):
        """Verify transitions figure generation."""
        trans = [
            BehaviourTransition(CLASS_READING_WRITING, CLASS_LOOKING_AWAY, 3, [10.0, 25.0, 40.0]),
            BehaviourTransition(CLASS_LOOKING_AWAY, CLASS_READING_WRITING, 2, [15.0, 30.0]),
        ]
        fig = create_transitions_figure(trans, track_id=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Empty transitions figure
        fig_empty = create_transitions_figure([], track_id=1)
        assert isinstance(fig_empty, plt.Figure)
        plt.close(fig_empty)

    def test_create_model_comparison_timeline_figure(self, sample_predictions_df):
        """Verify multi-model comparison timeline figure generation."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        segs = merge_behaviour_segments(t1_df)

        model_segs = {
            "RNN": segs,
            "LSTM": segs,
            "GRU": segs,
        }
        fig = create_model_comparison_timeline_figure(model_segs, track_id=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_classroom_overview_figure(self, sample_predictions_df):
        """Verify multi-track classroom overview timeline figure generation."""
        t1_df = extract_track_trajectory(sample_predictions_df, track_id=1)
        t2_df = extract_track_trajectory(sample_predictions_df, track_id=2)

        multi_tracks = {
            1: merge_behaviour_segments(t1_df),
            2: merge_behaviour_segments(t2_df),
        }
        fig = create_classroom_overview_figure(multi_tracks, model_name="LSTM")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
