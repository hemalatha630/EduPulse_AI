"""Unit Tests for Feature 7 — Temporal Sequence Creation.

Validates:
1. TemporalSequenceGenerator parameter validation.
2. Chronological sorting by extracted frame index.
3. Stride math and sliding window slicing.
4. Short track skipping and reporting.
5. Large tracking gap splitting policy.
6. Multi-track isolation (no feature mixing between different student tracks).
7. Behaviour class alignment and dominant behaviour calculation.
8. PyTorch tensor conversion and ClassroomSequenceDataset DataLoader compatibility.
9. End-to-end pipeline execution and file persistence (.npy and .csv).
10. Sequence timeline and track coverage visualizations.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import (
    CNN_FEATURES_NPY_FILENAME,
    CNN_METADATA_CSV_FILENAME,
    DEFAULT_MAX_FRAME_GAP,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEQUENCE_STRIDE,
    TEMPORAL_SEQUENCES_METADATA_FILENAME,
    TEMPORAL_SEQUENCES_NPY_FILENAME,
)
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


class TestTemporalSequenceGeneratorInit:
    """Test suite for TemporalSequenceGenerator configuration and validation."""

    def test_default_initialization(self):
        """Verify default parameters match config values."""
        generator = TemporalSequenceGenerator()
        assert generator.sequence_length == DEFAULT_SEQUENCE_LENGTH
        assert generator.stride == DEFAULT_SEQUENCE_STRIDE
        assert generator.max_frame_gap == DEFAULT_MAX_FRAME_GAP

    def test_custom_initialization(self):
        """Verify valid custom parameters are accepted."""
        generator = TemporalSequenceGenerator(sequence_length=8, stride=3, max_frame_gap=4)
        assert generator.sequence_length == 8
        assert generator.stride == 3
        assert generator.max_frame_gap == 4

    @pytest.mark.parametrize(
        "seq_len,stride,gap",
        [
            (0, 2, 2),
            (-5, 2, 2),
            (10, 0, 2),
            (10, -1, 2),
            (10, 2, 0),
            (10, 2, -3),
        ],
    )
    def test_invalid_parameters_raise_error(self, seq_len, stride, gap):
        """Verify non-positive parameter values raise ValueError."""
        with pytest.raises(ValueError):
            TemporalSequenceGenerator(sequence_length=seq_len, stride=stride, max_frame_gap=gap)


class TestTrackSequenceGeneration:
    """Test suite for sliding window logic and track processing."""

    @pytest.fixture
    def mock_track_data(self):
        """Generate synthetic observations for a single track (16 frames, 512-dim)."""
        num_frames = 16
        feature_dim = 512
        features = np.random.randn(num_frames, feature_dim).astype(np.float32)

        rows = []
        for i in range(num_frames):
            rows.append(
                {
                    "feature_index": i,
                    "video_id": "test_video",
                    "frame_id": 100 + i * 5,
                    "extracted_frame_index": i + 1,
                    "timestamp_seconds": round(i * 0.166, 3),
                    "track_id": 42,
                    "behaviour_class": "Attentive / Listening" if i < 10 else "Taking Notes / Writing",
                    "behaviour_confidence": 0.85 if i < 10 else 0.90,
                }
            )
        df = pd.DataFrame(rows)
        return df, features

    def test_sliding_window_count_and_shape(self, mock_track_data):
        """Verify window count matches formula (N - L) // stride + 1."""
        track_df, features = mock_track_data
        generator = TemporalSequenceGenerator(sequence_length=10, stride=2, max_frame_gap=2)

        seqs, metas, skipped = generator.generate_track_sequences(
            track_df=track_df,
            features_array=features,
            video_id="test_video",
        )

        # 16 frames, L=10, stride=2 -> (16 - 10) // 2 + 1 = 4 sequences
        assert len(seqs) == 4
        assert len(metas) == 4
        assert skipped == 0
        for seq in seqs:
            assert seq.shape == (10, 512)

    def test_chronological_sorting_integrity(self, mock_track_data):
        """Verify observations are sorted chronologically even if input df is shuffled."""
        track_df, features = mock_track_data
        # Shuffle dataframe rows
        shuffled_df = track_df.sample(frac=1.0, random_state=42).copy()

        generator = TemporalSequenceGenerator(sequence_length=10, stride=2)
        seqs, metas, _ = generator.generate_track_sequences(
            track_df=shuffled_df,
            features_array=features,
            video_id="test_video",
        )

        # Verify each sequence's frame indices are strictly increasing
        for meta in metas:
            assert meta["start_extracted_frame_index"] < meta["end_extracted_frame_index"]
            assert meta["start_timestamp_seconds"] < meta["end_timestamp_seconds"]

    def test_short_track_skipped_policy(self, mock_track_data):
        """Verify tracks with fewer observations than sequence_length are skipped."""
        track_df, features = mock_track_data
        short_df = track_df.iloc[:7]  # 7 frames < 10

        generator = TemporalSequenceGenerator(sequence_length=10, stride=2)
        seqs, metas, skipped = generator.generate_track_sequences(
            track_df=short_df,
            features_array=features,
            video_id="test_video",
        )

        assert len(seqs) == 0
        assert len(metas) == 0
        assert skipped == 1

    def test_frame_gap_splitting_policy(self):
        """Verify large tracking gaps split observations and prevent bridging."""
        # 20 observations total, but a gap from frame 10 to 25 (gap = 15 > max_frame_gap 2)
        feature_dim = 128
        features = np.random.randn(20, feature_dim).astype(np.float32)

        rows = []
        # Segment 1: 10 frames (indices 1..10)
        for i in range(10):
            rows.append(
                {
                    "feature_index": i,
                    "video_id": "gap_test",
                    "frame_id": i + 1,
                    "extracted_frame_index": i + 1,
                    "timestamp_seconds": i * 0.1,
                    "track_id": 5,
                    "behaviour_class": "Attentive / Listening",
                    "behaviour_confidence": 0.8,
                }
            )
        # Gap: jump from extracted_frame_index 10 to 25
        # Segment 2: 10 frames (indices 25..34)
        for i in range(10):
            rows.append(
                {
                    "feature_index": 10 + i,
                    "video_id": "gap_test",
                    "frame_id": 100 + i,
                    "extracted_frame_index": 25 + i,
                    "timestamp_seconds": 2.5 + i * 0.1,
                    "track_id": 5,
                    "behaviour_class": "Active Interaction",
                    "behaviour_confidence": 0.9,
                }
            )
        track_df = pd.DataFrame(rows)

        # With L=6, stride=2:
        # Segment 1 (10 frames): (10-6)//2 + 1 = 3 seqs
        # Segment 2 (10 frames): (10-6)//2 + 1 = 3 seqs
        # Total = 6 seqs (none bridging across the frame gap)
        generator = TemporalSequenceGenerator(sequence_length=6, stride=2, max_frame_gap=2)
        seqs, metas, skipped = generator.generate_track_sequences(
            track_df=track_df,
            features_array=features,
            video_id="gap_test",
        )

        assert len(seqs) == 6
        assert skipped == 0
        for meta in metas:
            # None of the sequences should cross the gap between frame 10 and 25
            assert not (meta["start_extracted_frame_index"] <= 10 and meta["end_extracted_frame_index"] >= 25)

    def test_dominant_behaviour_calculation(self, mock_track_data):
        """Verify dominant observable behaviour corresponds to window mode."""
        track_df, features = mock_track_data
        # First 10 frames are 'Attentive / Listening'
        generator = TemporalSequenceGenerator(sequence_length=10, stride=10)
        seqs, metas, _ = generator.generate_track_sequences(
            track_df=track_df,
            features_array=features,
            video_id="test_video",
        )

        assert metas[0]["dominant_behaviour"] == "Attentive / Listening"
        assert metas[0]["mean_behaviour_confidence"] == pytest.approx(0.85, rel=1e-2)


class TestMultipleTracksIsolation:
    """Test suite ensuring different students' tracks are never mixed."""

    def test_tracks_isolated_in_sequence_generation(self):
        """Verify each sequence contains features from exactly one track_id."""
        features = np.random.randn(30, 64).astype(np.float32)
        rows = []

        # Track 1: 15 frames
        for i in range(15):
            rows.append(
                {
                    "feature_index": i,
                    "video_id": "iso_test",
                    "frame_id": i + 1,
                    "extracted_frame_index": i + 1,
                    "timestamp_seconds": i * 0.1,
                    "track_id": 1,
                    "behaviour_class": "Attentive / Listening",
                    "behaviour_confidence": 0.8,
                }
            )
        # Track 2: 15 frames
        for i in range(15):
            rows.append(
                {
                    "feature_index": 15 + i,
                    "video_id": "iso_test",
                    "frame_id": i + 1,
                    "extracted_frame_index": i + 1,
                    "timestamp_seconds": i * 0.1,
                    "track_id": 2,
                    "behaviour_class": "Hand Raising",
                    "behaviour_confidence": 0.9,
                }
            )
        meta_df = pd.DataFrame(rows)

        generator = TemporalSequenceGenerator(sequence_length=8, stride=4)
        seqs_1, metas_1, _ = generator.generate_track_sequences(
            track_df=meta_df[meta_df["track_id"] == 1],
            features_array=features,
            video_id="iso_test",
        )
        seqs_2, metas_2, _ = generator.generate_track_sequences(
            track_df=meta_df[meta_df["track_id"] == 2],
            features_array=features,
            video_id="iso_test",
        )

        assert all(m["track_id"] == 1 for m in metas_1)
        assert all(m["track_id"] == 2 for m in metas_2)


class TestPyTorchDatasetAndTensor:
    """Test suite for PyTorch conversion and Dataset/DataLoader compatibility."""

    def test_sequences_to_tensor_conversion(self):
        """Verify (N, L, D) NumPy array converts to PyTorch FloatTensor."""
        np_arr = np.random.randn(8, 10, 512).astype(np.float32)
        tensor = sequences_to_tensor(np_arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (8, 10, 512)
        assert tensor.dtype == torch.float32

    def test_empty_sequences_to_tensor(self):
        """Verify empty array converts to empty 3D tensor."""
        empty_np = np.empty((0, 0, 0), dtype=np.float32)
        tensor = sequences_to_tensor(empty_np)
        assert tensor.shape == (0, 0, 0)

    def test_classroom_sequence_dataset_and_dataloader(self):
        """Verify ClassroomSequenceDataset supports PyTorch DataLoader batching."""
        np_arr = np.random.randn(12, 10, 256).astype(np.float32)
        meta_rows = [{"sequence_id": i, "track_id": 1} for i in range(12)]
        meta_df = pd.DataFrame(meta_rows)

        dataset = ClassroomSequenceDataset(sequences=np_arr, metadata_df=meta_df)
        assert len(dataset) == 12

        # Test index access
        sample_tensor, sample_meta = dataset[0]
        assert sample_tensor.shape == (10, 256)
        assert sample_meta["sequence_id"] == 0

        # Test DataLoader batching
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) == 3

        batch_tensors, _ = batches[0]
        assert batch_tensors.shape == (4, 10, 256)


class TestEndToEndPipeline:
    """Test suite for run_temporal_sequence_creation and file persistence."""

    def test_pipeline_missing_prerequisites(self, tmp_path):
        """Verify failure message when CNN feature files do not exist."""
        success, summary, meta_df, tensor_np, err = run_temporal_sequence_creation(
            video_id="nonexistent_video",
            cnn_features_path=tmp_path / "missing.npy",
            cnn_metadata_path=tmp_path / "missing.csv",
        )
        assert success is False
        assert "CNN features are required" in err

    def test_pipeline_success_and_file_creation(self, tmp_path):
        """Verify complete pipeline execution, summary metrics, and file output."""
        # Create synthetic CNN features and metadata
        features = np.random.randn(30, 512).astype(np.float32)
        rows = []
        for i in range(30):
            rows.append(
                {
                    "feature_index": i,
                    "video_id": "synth_vid",
                    "frame_id": i + 1,
                    "extracted_frame_index": i + 1,
                    "timestamp_seconds": i * 0.166,
                    "track_id": 1 if i < 15 else 2,
                    "behaviour_class": "Attentive / Listening",
                    "behaviour_confidence": 0.88,
                }
            )
        meta_df = pd.DataFrame(rows)

        cnn_npy_path = tmp_path / CNN_FEATURES_NPY_FILENAME
        cnn_csv_path = tmp_path / CNN_METADATA_CSV_FILENAME
        np.save(str(cnn_npy_path), features)
        meta_df.to_csv(cnn_csv_path, index=False)

        # Patch PROCESSED_DIR destination for this test
        import src.temporal.sequence_generator as seq_gen_module
        orig_processed = seq_gen_module.PROCESSED_DIR
        seq_gen_module.PROCESSED_DIR = tmp_path

        try:
            success, summary, out_df, out_npy, err = run_temporal_sequence_creation(
                video_id="synth_vid",
                cnn_features_path=cnn_npy_path,
                cnn_metadata_path=cnn_csv_path,
                sequence_length=10,
                stride=2,
            )

            assert success is True
            assert err == ""
            assert summary is not None
            assert summary.total_tracks_evaluated == 2
            assert summary.valid_tracks_processed == 2
            assert summary.sequence_length == 10
            assert summary.feature_dimension == 512
            assert summary.tensor_shape[1:] == (10, 512)

            # Check files were created
            assert (tmp_path / "synth_vid" / TEMPORAL_SEQUENCES_NPY_FILENAME).exists()
            assert (tmp_path / "synth_vid" / TEMPORAL_SEQUENCES_METADATA_FILENAME).exists()

            # Verify saved array matches returned array
            saved_npy = np.load(str(tmp_path / "synth_vid" / TEMPORAL_SEQUENCES_NPY_FILENAME))
            assert np.array_equal(saved_npy, out_npy)
        finally:
            seq_gen_module.PROCESSED_DIR = orig_processed


class TestTemporalVisualizations:
    """Test suite for timeline and coverage figures."""

    def test_create_sequence_timeline_figure(self):
        """Verify timeline figure generates without error and contains valid artist objects."""
        seq_features = np.random.randn(10, 512).astype(np.float32)
        seq_meta = {
            "sequence_id": 3,
            "track_id": 7,
            "start_timestamp_seconds": 1.5,
            "end_timestamp_seconds": 3.0,
            "frame_indices": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "behaviour_sequence": "Attentive / Listening -> Hand Raising",
            "dominant_behaviour": "Attentive / Listening",
        }

        fig = create_sequence_timeline_figure(seq_features, seq_meta)
        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert "Sequence #3" in ax.get_title()

    def test_create_track_coverage_figure(self):
        """Verify track coverage figure generates bars for metadata rows."""
        df = pd.DataFrame(
            [
                {
                    "track_id": 1,
                    "start_timestamp_seconds": 0.0,
                    "duration_seconds": 1.5,
                    "dominant_behaviour": "Attentive / Listening",
                },
                {
                    "track_id": 2,
                    "start_timestamp_seconds": 0.5,
                    "duration_seconds": 1.8,
                    "dominant_behaviour": "Taking Notes / Writing",
                },
            ]
        )

        fig = create_track_coverage_figure(df)
        assert fig is not None
        assert len(fig.axes) == 1

    def test_create_track_coverage_figure_empty_returns_none(self):
        """Verify empty metadata DataFrame returns None."""
        empty_df = pd.DataFrame()
        fig = create_track_coverage_figure(empty_df)
        assert fig is None
