"""Unit and Integration Test Suite for Feature 10: Teaching Activity Analysis."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.activity import (
    ACTIVITY_DISCUSSION,
    ACTIVITY_LECTURE,
    ACTIVITY_PRESENTATION,
    ACTIVITY_PROBLEM_SOLVING,
    ACTIVITY_SOURCE_MANUAL,
    ACTIVITY_UNKNOWN,
    TARGET_TEACHING_ACTIVITIES,
    TeachingActivitySegment,
    calculate_activity_behaviour_distributions,
    calculate_activity_transitions,
    create_activity_behaviour_distribution_figure,
    create_activity_behaviour_heatmap_figure,
    create_activity_timeline_figure,
    create_default_demo_segments,
    create_track_activity_figure,
    export_teaching_activity_results,
    generate_activity_summary_table,
    get_activity_description,
    get_activity_hex,
    get_activity_rgb,
    load_teaching_activity_segments,
    map_predictions_to_activities,
    save_teaching_activity_segments,
    validate_activity_segments,
)
from src.behaviour.behaviour_labels import (
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_READING_WRITING,
    TARGET_BEHAVIOUR_CLASSES,
)


def test_activity_labels_and_metadata():
    """Verify teaching activity constants, hex colors, and pedagogical descriptions."""
    assert len(TARGET_TEACHING_ACTIVITIES) == 4
    assert ACTIVITY_LECTURE in TARGET_TEACHING_ACTIVITIES
    assert ACTIVITY_DISCUSSION in TARGET_TEACHING_ACTIVITIES
    assert ACTIVITY_PROBLEM_SOLVING in TARGET_TEACHING_ACTIVITIES
    assert ACTIVITY_PRESENTATION in TARGET_TEACHING_ACTIVITIES

    # Colors
    assert get_activity_hex(ACTIVITY_LECTURE).startswith("#")
    assert len(get_activity_rgb(ACTIVITY_DISCUSSION)) == 3
    assert get_activity_hex("NonExistentActivity") == "#7F8C8D"

    # Descriptions
    desc = get_activity_description(ACTIVITY_LECTURE)
    assert "Instructor-led" in desc


def test_teaching_activity_segment_properties():
    """Verify segment dataclass formatting and dictionary conversion."""
    seg = TeachingActivitySegment(
        segment_id=1,
        video_id="test_vid",
        activity_class=ACTIVITY_LECTURE,
        start_timestamp_seconds=65.0,
        end_timestamp_seconds=130.0,
        duration_seconds=65.0,
        annotation_source=ACTIVITY_SOURCE_MANUAL,
    )
    assert seg.start_time_formatted == "01:05"
    assert seg.end_time_formatted == "02:10"
    d = seg.to_dict()
    assert d["start_time_formatted"] == "01:05"
    assert d["activity_class"] == ACTIVITY_LECTURE


def test_validate_activity_segments_success():
    """Valid non-overlapping segments pass validation without exception."""
    segments = [
        TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, 0.0, 10.0, 10.0),
        TeachingActivitySegment(2, "v1", ACTIVITY_DISCUSSION, 10.0, 20.0, 10.0),
    ]
    validate_activity_segments(segments, video_duration_seconds=20.0)


def test_validate_activity_segments_negative_start():
    """Segments with negative start timestamps fail validation."""
    segments = [TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, -1.0, 5.0, 6.0)]
    with pytest.raises(ValueError, match="negative"):
        validate_activity_segments(segments)


def test_validate_activity_segments_inverted_times():
    """Segments where start >= end timestamp fail validation."""
    segments = [TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, 10.0, 5.0, -5.0)]
    with pytest.raises(ValueError, match="precede end time"):
        validate_activity_segments(segments)


def test_validate_activity_segments_exceed_duration():
    """Segments exceeding video duration fail validation."""
    segments = [TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, 0.0, 30.0, 30.0)]
    with pytest.raises(ValueError, match="exceeds video duration"):
        validate_activity_segments(segments, video_duration_seconds=20.0)


def test_validate_activity_segments_invalid_activity():
    """Segments with unrecognized activity class fail validation."""
    segments = [TeachingActivitySegment(1, "v1", "MentalConcentration", 0.0, 10.0, 10.0)]
    with pytest.raises(ValueError, match="invalid activity"):
        validate_activity_segments(segments)


def test_validate_activity_segments_overlapping():
    """Overlapping segments raise a descriptive ValueError."""
    segments = [
        TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, 0.0, 12.0, 12.0),
        TeachingActivitySegment(2, "v1", ACTIVITY_DISCUSSION, 10.0, 20.0, 10.0),
    ]
    with pytest.raises(ValueError, match="Overlapping activity segments detected"):
        validate_activity_segments(segments)


def test_save_and_load_teaching_activity_segments(tmp_path: Path):
    """Test persistence and roundtrip loading of activity segments."""
    segments = [
        TeachingActivitySegment(1, "demo", ACTIVITY_LECTURE, 0.0, 15.0, 15.0),
        TeachingActivitySegment(2, "demo", ACTIVITY_DISCUSSION, 15.0, 30.0, 15.0),
    ]

    saved_df = save_teaching_activity_segments(
        video_id="demo",
        segments=segments,
        processed_dir=tmp_path,
        video_duration_seconds=30.0,
    )
    assert len(saved_df) == 2
    assert (tmp_path / "demo" / "teaching_activity_segments.csv").exists()

    loaded = load_teaching_activity_segments(video_id="demo", processed_dir=tmp_path)
    assert len(loaded) == 2
    assert loaded[0].activity_class == ACTIVITY_LECTURE
    assert loaded[1].activity_class == ACTIVITY_DISCUSSION
    assert loaded[1].start_timestamp_seconds == 15.0


def test_create_default_demo_segments():
    """Verify demo segment generation partitions the video cleanly without overlap."""
    segs_short = create_default_demo_segments(video_id="short_vid", video_duration_seconds=3.0)
    assert len(segs_short) == 2
    validate_activity_segments(segs_short, video_duration_seconds=3.0)

    segs_long = create_default_demo_segments(video_id="long_vid", video_duration_seconds=10.0)
    assert len(segs_long) == 3
    validate_activity_segments(segs_long, video_duration_seconds=10.0)


def test_map_predictions_to_activities():
    """Verify chronological assignment of activity segments to prediction rows."""
    segments = [
        TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, 0.0, 10.0, 10.0),
        TeachingActivitySegment(2, "v1", ACTIVITY_DISCUSSION, 10.0, 20.0, 10.0),
    ]

    preds = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "start_timestamp_seconds": [1.0, 11.0, 4.0, 22.0],
            "end_timestamp_seconds": [2.0, 12.0, 5.0, 23.0],
            "predicted_label": [
                CLASS_LOOKING_TOWARD_INSTRUCTION,
                CLASS_INTERACTING_WITH_PEERS,
                CLASS_READING_WRITING,
                CLASS_READING_WRITING,
            ],
        }
    )

    mapped = map_predictions_to_activities(preds, segments)
    assert "activity_class" in mapped.columns
    assert mapped["activity_class"].iloc[0] == ACTIVITY_LECTURE
    assert mapped["activity_class"].iloc[1] == ACTIVITY_DISCUSSION
    assert mapped["activity_class"].iloc[2] == ACTIVITY_LECTURE
    # Point at 22.0s is beyond 20.0s -> Unannotated
    assert mapped["activity_class"].iloc[3] == ACTIVITY_UNKNOWN


def test_calculate_activity_behaviour_distributions():
    """Verify observed duration and percentage share calculations across activities."""
    segments = [
        TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, 0.0, 10.0, 10.0),
        TeachingActivitySegment(2, "v1", ACTIVITY_DISCUSSION, 10.0, 20.0, 10.0),
    ]

    preds = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "start_timestamp_seconds": [0.0, 2.0, 11.0],
            "end_timestamp_seconds": [2.0, 4.0, 13.0],
            "predicted_label": [
                CLASS_LOOKING_TOWARD_INSTRUCTION,
                CLASS_LOOKING_TOWARD_INSTRUCTION,
                CLASS_INTERACTING_WITH_PEERS,
            ],
        }
    )

    mapped = map_predictions_to_activities(preds, segments)
    dist = calculate_activity_behaviour_distributions(mapped, segments=segments)

    assert not dist.empty
    assert "activity_class" in dist.columns
    assert "percentage_share" in dist.columns

    # In Lecture: Listening was active from 0.0 to 4.0 -> 4.0s duration
    lecture_listening = dist[
        (dist["activity_class"] == ACTIVITY_LECTURE)
        & (dist["behaviour_class"] == CLASS_LOOKING_TOWARD_INSTRUCTION)
    ]
    assert not lecture_listening.empty
    assert lecture_listening["observed_duration_seconds"].iloc[0] == 4.0
    assert lecture_listening["percentage_share"].iloc[0] == 100.0

    # In Discussion: Interacting with peers was active
    disc_peers = dist[
        (dist["activity_class"] == ACTIVITY_DISCUSSION)
        & (dist["behaviour_class"] == CLASS_INTERACTING_WITH_PEERS)
    ]
    assert not disc_peers.empty
    assert disc_peers["observed_duration_seconds"].iloc[0] == 2.0
    assert disc_peers["percentage_share"].iloc[0] == 100.0


def test_generate_activity_summary_table_and_tie_detection():
    """Verify summary table generation and explicit tie detection."""
    segments = [
        TeachingActivitySegment(1, "v1", ACTIVITY_LECTURE, 0.0, 10.0, 10.0),
    ]

    # Create tied distribution
    dist_tied = pd.DataFrame(
        [
            {
                "activity_class": ACTIVITY_LECTURE,
                "behaviour_class": CLASS_LOOKING_TOWARD_INSTRUCTION,
                "observed_duration_seconds": 5.0,
                "percentage_share": 50.0,
                "segment_count": 1,
                "sequence_count": 5,
                "track_id": "classroom",
            },
            {
                "activity_class": ACTIVITY_LECTURE,
                "behaviour_class": CLASS_READING_WRITING,
                "observed_duration_seconds": 5.0,
                "percentage_share": 50.0,
                "segment_count": 1,
                "sequence_count": 5,
                "track_id": "classroom",
            },
        ]
    )

    summary = generate_activity_summary_table(dist_tied, segments)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["Activity"] == ACTIVITY_LECTURE
    assert bool(row["Is Tie"]) is True
    assert "(Tied)" in row["Dominant Behaviour"]


def test_calculate_activity_transitions():
    """Verify behaviour transitions are accounted within teaching activities."""
    preds = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "start_timestamp_seconds": [0.0, 2.0, 4.0],
            "end_timestamp_seconds": [2.0, 4.0, 6.0],
            "predicted_label": [
                CLASS_LOOKING_TOWARD_INSTRUCTION,
                CLASS_READING_WRITING,
                CLASS_LOOKING_TOWARD_INSTRUCTION,
            ],
            "activity_class": [ACTIVITY_LECTURE, ACTIVITY_LECTURE, ACTIVITY_LECTURE],
        }
    )

    trans = calculate_activity_transitions(preds)
    assert not trans.empty
    assert "transition_label" in trans.columns
    assert len(trans) == 2


def test_export_teaching_activity_results(tmp_path: Path):
    """Verify CSV export of activity results."""
    segments = [TeachingActivitySegment(1, "demo", ACTIVITY_LECTURE, 0.0, 10.0, 10.0)]
    summary_df = pd.DataFrame([{"Activity": ACTIVITY_LECTURE, "Total Observed (s)": 10.0}])
    dist_df = pd.DataFrame([{"activity_class": ACTIVITY_LECTURE, "percentage_share": 100.0}])
    trans_df = pd.DataFrame([{"activity_class": ACTIVITY_LECTURE, "count": 1}])

    paths = export_teaching_activity_results(
        video_id="demo",
        model_type="lstm",
        segments=segments,
        summary_df=summary_df,
        distribution_df=dist_df,
        transitions_df=trans_df,
        results_dir=tmp_path,
    )
    for p in paths.values():
        assert p.exists()


def test_visualizations():
    """Verify all 4 visualization functions produce valid Matplotlib figures without crashing."""
    segments = [
        TeachingActivitySegment(1, "demo", ACTIVITY_LECTURE, 0.0, 10.0, 10.0),
        TeachingActivitySegment(2, "demo", ACTIVITY_DISCUSSION, 10.0, 20.0, 10.0),
    ]
    dist_df = pd.DataFrame(
        [
            {
                "activity_class": ACTIVITY_LECTURE,
                "behaviour_class": CLASS_LOOKING_TOWARD_INSTRUCTION,
                "observed_duration_seconds": 8.0,
                "percentage_share": 80.0,
                "segment_count": 1,
                "sequence_count": 8,
                "track_id": "classroom",
            },
            {
                "activity_class": ACTIVITY_LECTURE,
                "behaviour_class": CLASS_READING_WRITING,
                "observed_duration_seconds": 2.0,
                "percentage_share": 20.0,
                "segment_count": 1,
                "sequence_count": 2,
                "track_id": "classroom",
            },
            {
                "activity_class": ACTIVITY_DISCUSSION,
                "behaviour_class": CLASS_INTERACTING_WITH_PEERS,
                "observed_duration_seconds": 10.0,
                "percentage_share": 100.0,
                "segment_count": 1,
                "sequence_count": 10,
                "track_id": "classroom",
            },
        ]
    )
    summary_df = pd.DataFrame(
        [
            {
                "Activity": ACTIVITY_LECTURE,
                "Dominant Behaviour": CLASS_LOOKING_TOWARD_INSTRUCTION,
            },
            {
                "Activity": ACTIVITY_DISCUSSION,
                "Dominant Behaviour": CLASS_INTERACTING_WITH_PEERS,
            },
        ]
    )

    # 1. Timeline
    fig1 = create_activity_timeline_figure(segments, video_duration_seconds=20.0, summary_df=summary_df)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # 2. Grouped Bar Distribution
    fig2 = create_activity_behaviour_distribution_figure(dist_df)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    # 3. Heatmap
    fig3 = create_activity_behaviour_heatmap_figure(dist_df)
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    # 4. Track Breakdown
    fig4 = create_track_activity_figure(dist_df, track_id=1)
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)
