"""Unit Tests for Feature 8 — RNN / LSTM / GRU Temporal Modelling.

Validates:
1. Observable behaviour label encoding and uncertain sample handling.
2. Track-grouped train/val/test splitting (strictly prevents data leakage).
3. Class weight calculation strictly from training data.
4. PyTorch recurrent models (Vanilla RNN, LSTM, GRU) forward passes.
5. Training loop, loss convergence, checkpoint persistence, and early stopping.
6. Fair multi-model training under identical experimental conditions.
7. Test set evaluation (Accuracy, Macro F1, Weighted F1, Confusion Matrix, Per-class metrics).
8. Fair model comparison table generation.
9. Single-sequence inference and batch predictions CSV export.
10. Visualization figures (training curves, confusion matrix, model comparison).
"""

from pathlib import Path
import tempfile
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import (
    CLASS_HEAD_DOWN,
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_READING_WRITING,
    CLASS_UNKNOWN,
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
)
from src.temporal.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_TARGET_CLASSES,
    SupervisedSequenceDataset,
    TARGET_CLASSES,
    compute_training_class_weights,
    create_split_dataloaders,
    encode_behaviour_labels,
    prepare_track_grouped_splits,
)
from src.temporal.evaluator import (
    compare_temporal_models,
    evaluate_temporal_model,
    generate_batch_predictions,
    predict_sequence,
)
from src.temporal.models import (
    BaseTemporalClassifier,
    GRUClassifier,
    LSTMClassifier,
    RNNClassifier,
    create_temporal_model,
)
from src.temporal.trainer import (
    TrainConfig,
    TrainResult,
    set_seed,
    train_all_temporal_models,
    train_temporal_model,
)
from src.temporal.visualization import (
    create_confusion_matrix_figure,
    create_model_comparison_figure,
    create_training_curves_figure,
)


@pytest.fixture
def synthetic_sequence_dataset():
    """Create synthetic sequences and metadata across 6 tracks and multiple classes."""
    num_tracks = 6
    seqs_per_track = 5
    total_seqs = num_tracks * seqs_per_track
    seq_len = 10
    feature_dim = 64

    np.random.seed(42)
    sequences = np.random.randn(total_seqs, seq_len, feature_dim).astype(np.float32)

    rows = []
    classes = TARGET_CLASSES[:4]  # Use 4 of the 6 classes for distribution
    for t_idx in range(num_tracks):
        track_id = 101 + t_idx
        track_class = classes[t_idx % len(classes)]
        for s_idx in range(seqs_per_track):
            global_id = t_idx * seqs_per_track + s_idx
            rows.append(
                {
                    "sequence_id": global_id,
                    "video_id": "synth_classroom",
                    "track_id": track_id,
                    "start_timestamp_seconds": float(s_idx * 1.5),
                    "end_timestamp_seconds": float(s_idx * 1.5 + 1.5),
                    "duration_seconds": 1.5,
                    "dominant_behaviour": track_class,
                    "mean_behaviour_confidence": 0.85,
                }
            )
    metadata_df = pd.DataFrame(rows)
    return sequences, metadata_df


class TestDatasetAndDataLeakagePrevention:
    """Test suite for label encoding, class weighting, and leakage prevention."""

    def test_encode_behaviour_labels(self):
        """Verify mapping from behaviour strings to valid integer indices."""
        df = pd.DataFrame(
            {
                "dominant_behaviour": [
                    CLASS_LOOKING_TOWARD_INSTRUCTION,
                    CLASS_READING_WRITING,
                    CLASS_INTERACTING_WITH_PEERS,
                    CLASS_UNKNOWN,  # Should be excluded
                ]
            }
        )
        valid_mask, labels, excluded = encode_behaviour_labels(df)
        assert np.array_equal(valid_mask, [True, True, True, False])
        assert len(labels) == 3
        assert labels[0] == CLASS_TO_IDX[CLASS_LOOKING_TOWARD_INSTRUCTION]
        assert labels[1] == CLASS_TO_IDX[CLASS_READING_WRITING]
        assert labels[2] == CLASS_TO_IDX[CLASS_INTERACTING_WITH_PEERS]
        assert excluded == 1

    def test_track_grouped_splits_strictly_prevents_leakage(self, synthetic_sequence_dataset):
        """CRITICAL: Verify sequences from the same track NEVER cross train/val/test splits."""
        seqs, meta_df = synthetic_sequence_dataset
        split = prepare_track_grouped_splits(
            sequences_array=seqs,
            metadata_df=meta_df,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            random_seed=42,
        )

        train_tracks = set(split.train_tracks)
        val_tracks = set(split.val_tracks)
        test_tracks = set(split.test_tracks)

        # Disjoint sets: strictly ZERO overlap
        assert len(train_tracks.intersection(val_tracks)) == 0
        assert len(train_tracks.intersection(test_tracks)) == 0
        assert len(val_tracks.intersection(test_tracks)) == 0

        # Verify all sequences belonging to a track are strictly in that track's split
        train_meta_tracks = set(split.train_dataset.metadata_df["track_id"].unique())
        val_meta_tracks = set(split.val_dataset.metadata_df["track_id"].unique())
        test_meta_tracks = set(split.test_dataset.metadata_df["track_id"].unique())

        assert train_meta_tracks == train_tracks
        assert val_meta_tracks == val_tracks
        assert test_meta_tracks == test_tracks

    def test_compute_training_class_weights_on_train_only(self):
        """Verify class weights are inversely proportional to class frequency on train labels."""
        # Class 0: 10 samples, Class 1: 2 samples, others: 0
        train_labels = np.array([0] * 10 + [1] * 2, dtype=np.int64)
        weights = compute_training_class_weights(train_labels, num_classes=NUM_TARGET_CLASSES)

        assert isinstance(weights, torch.Tensor)
        assert len(weights) == NUM_TARGET_CLASSES
        # Minority class 1 should have higher weight than majority class 0
        assert weights[1].item() > weights[0].item()
        # Unseen classes have neutral weight 1.0
        assert weights[2].item() == pytest.approx(1.0)

    def test_supervised_sequence_dataset_and_dataloader(self, synthetic_sequence_dataset):
        """Verify SupervisedSequenceDataset yields tensors of correct shape and dtype."""
        seqs, meta_df = synthetic_sequence_dataset
        split = prepare_track_grouped_splits(seqs, meta_df)
        train_loader, val_loader, test_loader = create_split_dataloaders(split, batch_size=4)

        for batch_x, batch_y, batch_meta in train_loader:
            assert batch_x.ndim == 3
            assert batch_x.shape[1] == 10  # seq_len
            assert batch_x.shape[2] == 64  # feature_dim
            assert batch_y.dtype == torch.int64
            assert "track_id" in batch_meta
            break


class TestTemporalModelArchitectures:
    """Test suite for RNN, LSTM, and GRU model architectures."""

    @pytest.mark.parametrize("model_type", [MODEL_RNN, MODEL_LSTM, MODEL_GRU])
    def test_model_forward_pass_and_shape(self, model_type):
        """Verify all three models accept (B, L, D) and output logits (B, C)."""
        batch_size = 4
        seq_len = 10
        feature_dim = 64
        num_classes = 6

        model = create_temporal_model(
            model_type=model_type,
            input_size=feature_dim,
            hidden_size=32,
            num_layers=1,
            dropout=0.2,
            num_classes=num_classes,
        )

        dummy_input = torch.randn(batch_size, seq_len, feature_dim)
        logits = model(dummy_input)

        assert logits.shape == (batch_size, num_classes)
        assert not torch.isnan(logits).any()

    def test_parameter_counting(self):
        """Verify parameter counter returns valid dictionary."""
        model = LSTMClassifier(input_size=64, hidden_size=32, num_layers=1)
        params = model.count_parameters()
        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0
        assert params["trainable"] == params["total"]

    def test_invalid_model_type_raises_error(self):
        """Verify invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model_type"):
            create_temporal_model(model_type="invalid_transformer_type")


class TestTemporalTrainingEngine:
    """Test suite for training loop, checkpointing, and early stopping."""

    @pytest.mark.parametrize("m_type", [MODEL_RNN, MODEL_LSTM, MODEL_GRU])
    def test_train_temporal_model(self, m_type, synthetic_sequence_dataset, tmp_path):
        """Verify training loop runs, loss is computed, and checkpoint is saved for all models."""
        seqs, meta_df = synthetic_sequence_dataset
        split = prepare_track_grouped_splits(seqs, meta_df, random_seed=42)

        config = TrainConfig(
            model_type=m_type,
            input_size=64,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            num_classes=NUM_TARGET_CLASSES,
            learning_rate=0.005,
            batch_size=4,
            epochs=3,
            patience=3,
            seed=42,
        )

        model, result = train_temporal_model(
            config=config,
            split=split,
            checkpoint_dir=tmp_path / "models",
            results_dir=tmp_path / "results",
        )

        assert isinstance(model, BaseTemporalClassifier)
        assert isinstance(result, TrainResult)
        assert result.model_type == m_type
        assert len(result.history_df) == 3
        assert result.checkpoint_path.exists()

        # Check saved checkpoint can be loaded
        ckpt = torch.load(str(result.checkpoint_path), weights_only=False)
        assert ckpt["model_type"] == m_type
        assert "model_state_dict" in ckpt
        assert "config" in ckpt

    def test_train_all_temporal_models_fair_comparison(self, synthetic_sequence_dataset, tmp_path):
        """Verify train_all_temporal_models trains RNN, LSTM, and GRU under identical config."""
        seqs, meta_df = synthetic_sequence_dataset
        split = prepare_track_grouped_splits(seqs, meta_df, random_seed=42)

        base_config = TrainConfig(
            input_size=64,
            hidden_size=32,
            epochs=2,
            batch_size=4,
            patience=2,
            seed=42,
        )

        results = train_all_temporal_models(
            base_config=base_config,
            split=split,
            checkpoint_dir=tmp_path / "models",
            results_dir=tmp_path / "results",
        )

        assert set(results.keys()) == {MODEL_RNN, MODEL_LSTM, MODEL_GRU}
        for m_type, (model, res) in results.items():
            assert res.model_type == m_type
            assert res.checkpoint_path.exists()


class TestEvaluationAndInference:
    """Test suite for test-set evaluation, comparison table, and inference."""

    @pytest.fixture
    def trained_lstm_setup(self, synthetic_sequence_dataset, tmp_path):
        """Train a lightweight LSTM model on synthetic data for evaluation testing."""
        seqs, meta_df = synthetic_sequence_dataset
        split = prepare_track_grouped_splits(seqs, meta_df, random_seed=42)

        config = TrainConfig(
            model_type=MODEL_LSTM,
            input_size=64,
            hidden_size=32,
            epochs=3,
            batch_size=4,
            seed=42,
        )

        model, res = train_temporal_model(
            config=config,
            split=split,
            checkpoint_dir=tmp_path / "models",
            results_dir=tmp_path / "results",
        )
        return model, split, tmp_path

    def test_evaluate_temporal_model(self, trained_lstm_setup):
        """Verify test set evaluation produces all required metrics."""
        model, split, tmp_path = trained_lstm_setup
        _, _, test_loader = create_split_dataloaders(split, batch_size=4)

        metrics = evaluate_temporal_model(
            model=model,
            test_loader=test_loader,
            results_dir=tmp_path / "results" / "lstm",
        )

        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert 0.0 <= metrics.weighted_f1 <= 1.0
        assert 0.0 <= metrics.macro_precision <= 1.0
        assert 0.0 <= metrics.macro_recall <= 1.0
        assert metrics.confusion_matrix.shape == (NUM_TARGET_CLASSES, NUM_TARGET_CLASSES)
        assert len(metrics.per_class_metrics) == NUM_TARGET_CLASSES
        assert (tmp_path / "results" / "lstm" / "metrics.json").exists()

    def test_compare_temporal_models_table(self, trained_lstm_setup):
        """Verify side-by-side model comparison DataFrame."""
        model, split, tmp_path = trained_lstm_setup
        _, _, test_loader = create_split_dataloaders(split, batch_size=4)

        metrics = evaluate_temporal_model(model=model, test_loader=test_loader)
        comp_df = compare_temporal_models({MODEL_LSTM: metrics})

        assert isinstance(comp_df, pd.DataFrame)
        assert "Model" in comp_df.columns
        assert "Accuracy" in comp_df.columns
        assert "Macro F1" in comp_df.columns
        assert "Weighted F1" in comp_df.columns
        assert comp_df.iloc[0]["Model"] == "LSTM"

    def test_predict_single_sequence(self, trained_lstm_setup):
        """Verify single sequence inference returns top class, confidence, and prob distribution."""
        model, _, _ = trained_lstm_setup
        single_seq = np.random.randn(10, 64).astype(np.float32)

        pred_class, conf, probs = predict_sequence(model, single_seq)

        assert pred_class in TARGET_CLASSES
        assert 0.0 <= conf <= 1.0
        assert len(probs) == NUM_TARGET_CLASSES
        assert pytest.approx(sum(probs.values()), rel=1e-2) == 1.0

    def test_generate_batch_predictions(self, trained_lstm_setup):
        """Verify predictions.csv is created with expected schema."""
        model, split, tmp_path = trained_lstm_setup
        out_csv = tmp_path / "results" / "predictions.csv"

        df = generate_batch_predictions(
            model=model,
            dataset=split.test_dataset,
            video_id="synth_classroom",
            output_csv_path=out_csv,
        )

        assert out_csv.exists()
        assert len(df) == len(split.test_dataset)
        expected_cols = [
            "model",
            "video_id",
            "track_id",
            "sequence_id",
            "start_timestamp_seconds",
            "end_timestamp_seconds",
            "true_label",
            "predicted_label",
            "confidence",
        ]
        for col in expected_cols:
            assert col in df.columns


class TestVisualizations:
    """Test suite for training curves, confusion matrix, and model comparison figures."""

    def test_create_training_curves_figure(self):
        """Verify training loss and accuracy curves figure."""
        history_df = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [1.5, 1.2, 0.9],
                "train_acc": [0.4, 0.6, 0.75],
                "val_loss": [1.4, 1.1, 0.85],
                "val_acc": [0.45, 0.62, 0.78],
            }
        )
        fig = create_training_curves_figure(history_df, "LSTM")
        assert fig is not None
        assert len(fig.axes) == 2

    def test_create_confusion_matrix_figure(self):
        """Verify confusion matrix heatmap figure."""
        cm = np.zeros((NUM_TARGET_CLASSES, NUM_TARGET_CLASSES), dtype=int)
        cm[0, 0] = 5
        cm[1, 1] = 8
        fig = create_confusion_matrix_figure(cm, TARGET_CLASSES, "LSTM")
        assert fig is not None
        assert len(fig.axes) == 2  # image ax + colorbar ax

    def test_create_model_comparison_figure(self):
        """Verify model comparison bar chart figure."""
        df = pd.DataFrame(
            [
                {"Model": "RNN", "Accuracy": "65.0%", "Macro F1": "0.62", "Weighted F1": "0.64"},
                {"Model": "LSTM", "Accuracy": "78.0%", "Macro F1": "0.76", "Weighted F1": "0.77"},
                {"Model": "GRU", "Accuracy": "76.0%", "Macro F1": "0.74", "Weighted F1": "0.75"},
            ]
        )
        fig = create_model_comparison_figure(df)
        assert fig is not None
        assert len(fig.axes) == 1
