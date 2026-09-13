"""Unit Tests for Feature 6 — CNN Visual Feature Extraction.

Validates:
1. CNN model loading (ResNet18, weights, CPU/CUDA device auto-detection).
2. Single-crop feature vector extraction (512-dim).
3. Batched feature vector extraction (B, 512).
4. Safe handling of empty/invalid/out-of-bounds crops.
5. End-to-end extraction pipeline on video frames and tracking data.
6. Preservation of spatial-temporal ordering and behaviour linking.
7. 2D PCA feature space projection and figure generation.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.cnn.feature_extractor import (
    CNNFeatureExtractor,
    ExtractionSummary,
    run_cnn_feature_extraction,
)
from src.cnn.visualization import (
    compute_pca_2d,
    create_pca_scatter_figure,
)
from src.config import (
    CNN_FEATURE_DIM,
    CNN_FEATURES_NPY_FILENAME,
    CNN_METADATA_CSV_FILENAME,
    DEFAULT_CNN_MODEL,
    PROCESSED_DIR,
)


@pytest.fixture(scope="module")
def shared_feature_extractor():
    """Module-level fixture to load ResNet18 once across tests for speed."""
    extractor = CNNFeatureExtractor(model_name="resnet18")
    return extractor


class TestCNNModelLoading:
    """Test suite for CNN model initialization and device detection."""

    def test_resnet18_initialization(self, shared_feature_extractor):
        """Verify ResNet18 backbone initializes with stripped classification layer."""
        extractor = shared_feature_extractor
        assert extractor.model is not None
        assert extractor.feature_dim == 512
        assert extractor.device_name in ["CPU", "CUDA"]
        assert isinstance(extractor.model.fc, torch.nn.Identity)

    def test_unsupported_model_raises_error(self):
        """Verify invalid model names raise a ValueError."""
        with pytest.raises(ValueError, match="Unsupported CNN model"):
            CNNFeatureExtractor(model_name="unsupported_vgg99")


class TestFeatureVectorExtraction:
    """Test suite for single and batched tensor extraction."""

    def test_single_crop_extraction(self, shared_feature_extractor):
        """Verify single normalized crop tensor produces a (512,) float32 vector."""
        extractor = shared_feature_extractor
        crop_tensor = torch.randn(1, 3, 224, 224)
        feature_vec = extractor.extract_single(crop_tensor)

        assert isinstance(feature_vec, np.ndarray)
        assert feature_vec.shape == (512,)
        assert feature_vec.dtype == np.float32
        assert not np.isnan(feature_vec).any()
        assert not np.isinf(feature_vec).any()

    def test_single_crop_extraction_3d_input(self, shared_feature_extractor):
        """Verify 3D tensor (3, 224, 224) is auto-expanded to (1, 3, 224, 224)."""
        extractor = shared_feature_extractor
        crop_tensor = torch.randn(3, 224, 224)
        feature_vec = extractor.extract_single(crop_tensor)

        assert feature_vec.shape == (512,)

    def test_batch_crops_extraction(self, shared_feature_extractor):
        """Verify batch tensor (B, 3, 224, 224) produces (B, 512) feature matrix."""
        extractor = shared_feature_extractor
        batch_size = 5
        batch_tensor = torch.randn(batch_size, 3, 224, 224)
        features = extractor.extract_batch(batch_tensor)

        assert isinstance(features, np.ndarray)
        assert features.shape == (batch_size, 512)
        assert features.dtype == np.float32
        assert not np.isnan(features).any()

    def test_empty_batch_crops_extraction(self, shared_feature_extractor):
        """Verify empty batch returns empty (0, 512) array without error."""
        extractor = shared_feature_extractor
        empty_tensor = torch.empty(0, 3, 224, 224)
        features = extractor.extract_batch(empty_tensor)

        assert isinstance(features, np.ndarray)
        assert features.shape == (0, 512)


class TestFeatureExtractionPipeline:
    """Test suite for end-to-end extraction across frames and tracks."""

    @pytest.fixture
    def mock_video_data(self, tmp_path, monkeypatch):
        """Set up mock frames and tracking directory structure."""
        video_id = "test_cnn_mock_vid"
        frames_dir = tmp_path / "frames" / video_id
        processed_dir = tmp_path / "processed" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Patch paths in src.cnn.feature_extractor
        monkeypatch.setattr("src.cnn.feature_extractor.FRAMES_DIR", tmp_path / "frames")
        monkeypatch.setattr("src.cnn.feature_extractor.PROCESSED_DIR", tmp_path / "processed")

        # Create 3 synthetic frame images (720x1280x3)
        frame_meta_list = []
        track_meta_list = []
        beh_meta_list = []

        for f_idx in range(1, 4):
            filename = f"frame_{f_idx:06d}.jpg"
            img_path = frames_dir / filename
            # Create synthetic RGB frame with some texture
            synthetic_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            synthetic_img[100:400, 200:500] = [180, 120, 90]  # Student 1 region
            synthetic_img[150:450, 600:900] = [90, 150, 180]  # Student 2 region
            cv2.imwrite(str(img_path), synthetic_img)

            frame_meta_list.append({
                "video_id": video_id,
                "frame_id": (f_idx - 1) * 5,
                "extracted_frame_index": f_idx,
                "timestamp_seconds": round((f_idx - 1) * 0.167, 3),
                "frame_filename": filename,
            })

            # Track 1: Valid bbox
            track_meta_list.append({
                "video_id": video_id,
                "frame_id": (f_idx - 1) * 5,
                "extracted_frame_index": f_idx,
                "timestamp_seconds": round((f_idx - 1) * 0.167, 3),
                "frame_filename": filename,
                "track_id": 1,
                "confidence": 0.88,
                "x1": 200.0,
                "y1": 100.0,
                "x2": 500.0,
                "y2": 400.0,
            })

            # Track 2: Valid bbox
            track_meta_list.append({
                "video_id": video_id,
                "frame_id": (f_idx - 1) * 5,
                "extracted_frame_index": f_idx,
                "timestamp_seconds": round((f_idx - 1) * 0.167, 3),
                "frame_filename": filename,
                "track_id": 2,
                "confidence": 0.79,
                "x1": 600.0,
                "y1": 150.0,
                "x2": 900.0,
                "y2": 450.0,
            })

            # Track 3: Invalid / tiny crop (width=5, height=5) -> should be skipped safely
            track_meta_list.append({
                "video_id": video_id,
                "frame_id": (f_idx - 1) * 5,
                "extracted_frame_index": f_idx,
                "timestamp_seconds": round((f_idx - 1) * 0.167, 3),
                "frame_filename": filename,
                "track_id": 3,
                "confidence": 0.35,
                "x1": 10.0,
                "y1": 10.0,
                "x2": 15.0,
                "y2": 15.0,
            })

            # Linked behaviour data
            beh_meta_list.append({
                "video_id": video_id,
                "frame_id": (f_idx - 1) * 5,
                "extracted_frame_index": f_idx,
                "timestamp_seconds": round((f_idx - 1) * 0.167, 3),
                "frame_filename": filename,
                "track_id": 1,
                "behaviour_class": "Looking toward the instructional activity",
                "confidence": 0.85,
            })
            beh_meta_list.append({
                "video_id": video_id,
                "frame_id": (f_idx - 1) * 5,
                "extracted_frame_index": f_idx,
                "timestamp_seconds": round((f_idx - 1) * 0.167, 3),
                "frame_filename": filename,
                "track_id": 2,
                "behaviour_class": "Reading/writing",
                "confidence": 0.78,
            })

        frames_df = pd.DataFrame(frame_meta_list)
        tracks_df = pd.DataFrame(track_meta_list)
        behaviours_df = pd.DataFrame(beh_meta_list)

        return video_id, frames_df, tracks_df, behaviours_df, processed_dir

    def test_run_feature_extraction_pipeline(self, shared_feature_extractor, mock_video_data):
        """Verify full pipeline extracts features, skips invalid crops, and saves artifacts."""
        video_id, frames_df, tracks_df, behaviours_df, processed_dir = mock_video_data
        extractor = shared_feature_extractor

        progress_calls = []

        def dummy_cb(progress, msg):
            progress_calls.append((progress, msg))

        success, summary, meta_df, features_arr, err_msg = run_cnn_feature_extraction(
            video_id=video_id,
            frames_df=frames_df,
            tracks_df=tracks_df,
            feature_extractor=extractor,
            behaviours_df=behaviours_df,
            batch_size=2,
            frame_selection_mode="all",
            progress_callback=dummy_cb,
        )

        assert success is True, f"Extraction failed: {err_msg}"
        assert err_msg == ""
        assert summary is not None
        assert meta_df is not None
        assert features_arr is not None

        # 3 frames * 2 valid tracks = 6 valid features (Track 3 is skipped)
        assert summary.total_crops_evaluated == 9
        assert summary.valid_features_extracted == 6
        assert summary.skipped_crops_count == 3
        assert features_arr.shape == (6, 512)
        assert len(meta_df) == 6

        # Check progress callback was triggered
        assert len(progress_calls) > 0

        # Check files on disk
        npy_file = processed_dir / CNN_FEATURES_NPY_FILENAME
        csv_file = processed_dir / CNN_METADATA_CSV_FILENAME

        assert npy_file.exists()
        assert csv_file.exists()

        # Check saved numpy array can be reloaded
        loaded_npy = np.load(str(npy_file))
        assert loaded_npy.shape == (6, 512)
        np.testing.assert_allclose(loaded_npy, features_arr, rtol=1e-5)

        # Check metadata columns
        expected_cols = {
            "video_id",
            "frame_id",
            "extracted_frame_index",
            "timestamp_seconds",
            "frame_filename",
            "track_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "feature_index",
            "feature_path",
            "behaviour_class",
            "behaviour_confidence",
        }
        assert expected_cols.issubset(set(meta_df.columns))

        # Check feature_index indexing matches row order 0..N-1
        assert meta_df["feature_index"].tolist() == list(range(6))

        # Check behaviour label linking
        track_1_rows = meta_df[meta_df["track_id"] == 1]
        assert (track_1_rows["behaviour_class"] == "Looking toward the instructional activity").all()

        track_2_rows = meta_df[meta_df["track_id"] == 2]
        assert (track_2_rows["behaviour_class"] == "Reading/writing").all()

    def test_empty_frames_or_tracks_fail_gracefully(self, shared_feature_extractor):
        """Verify empty inputs return clear descriptive error message without crashing."""
        extractor = shared_feature_extractor
        empty_df = pd.DataFrame()

        success, summary, _, _, err = run_cnn_feature_extraction(
            video_id="dummy",
            frames_df=empty_df,
            tracks_df=empty_df,
            feature_extractor=extractor,
        )
        assert success is False
        assert summary is None
        assert "empty" in err.lower()


class TestPCAVisualization:
    """Test suite for 2D PCA projection and plotting."""

    def test_compute_pca_2d_valid_features(self):
        """Verify 2D PCA projects (N, 512) down to (N, 2) with positive variance."""
        np.random.seed(42)
        features = np.random.randn(20, 512).astype(np.float32)
        projected, var_explained = compute_pca_2d(features)

        assert projected is not None
        assert projected.shape == (20, 2)
        assert var_explained[0] >= 0.0
        assert var_explained[1] >= 0.0
        assert var_explained[0] + var_explained[1] <= 1.05

    def test_compute_pca_2d_insufficient_samples(self):
        """Verify compute_pca_2d handles 1 or 0 samples safely."""
        features_single = np.random.randn(1, 512)
        projected, var_explained = compute_pca_2d(features_single)
        assert projected is None
        assert var_explained == (0.0, 0.0)

    def test_create_pca_scatter_figure(self):
        """Verify Matplotlib scatter figure generates properly for Streamlit."""
        np.random.seed(42)
        projected = np.random.randn(12, 2)
        metadata_df = pd.DataFrame({
            "track_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "behaviour_class": [
                "Reading/writing",
                "Looking toward the instructional activity",
                "Interacting with peers",
            ] * 4,
        })

        # Test color by track_id
        fig1 = create_pca_scatter_figure(projected, metadata_df, color_by="track_id")
        assert fig1 is not None

        # Test color by behaviour_class
        fig2 = create_pca_scatter_figure(projected, metadata_df, color_by="behaviour_class")
        assert fig2 is not None
