"""Unit and Integration Test Suite for Feature 11: Research Experiments, Ablation, and Temporal Error Analysis."""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.activity.manager import TeachingActivitySegment
from src.config import (
    CLASS_HEAD_DOWN,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_READING_WRITING,
    CNN_FEATURE_DIM,
    MODEL_FRAME_CNN,
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
)
from src.experiments.ablation import (
    AblationTrial,
    run_all_ablation_studies,
    run_architecture_ablation,
    run_loss_weighting_ablation,
)
from src.experiments.baseline import (
    BaselineComparisonSummary,
    FrameCNNClassifier,
    run_baseline_comparison,
    train_frame_cnn_baseline,
)
from src.experiments.error_analysis import (
    TemporalErrorAnalysisSummary,
    TemporalErrorRecord,
    analyze_temporal_errors,
    categorize_duration_tier,
    compute_track_transition_distances,
    is_visually_similar,
)
from src.experiments.reporting import (
    FORBIDDEN_TERMS,
    generate_per_class_table,
    identify_top_confused_classes,
    synthesize_research_conclusions,
)
from src.experiments.visualization import (
    plot_ablation_summary,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_per_class_f1,
    plot_temporal_error_dynamics,
    plot_training_curves,
)
from src.temporal.dataset import (
    CLASS_TO_IDX,
    NUM_TARGET_CLASSES,
    SupervisedSequenceDataset,
    TARGET_CLASSES,
    TemporalDatasetSplit,
    create_split_dataloaders,
    prepare_track_grouped_splits,
)
from src.temporal.evaluator import EvaluationMetrics, evaluate_temporal_model
from src.temporal.models import GRUClassifier, LSTMClassifier, RNNClassifier
from src.temporal.trainer import TrainConfig, train_temporal_model


@pytest.fixture
def mock_dataset_split():
    """Create deterministic mock temporal sequence dataset split across 3 tracks."""
    np.random.seed(42)
    n_samples = 40
    seq_len = 10
    feat_dim = CNN_FEATURE_DIM

    seqs = np.random.randn(n_samples, seq_len, feat_dim).astype(np.float32)

    # Distribute samples across 3 tracks (Track 1 = Train, Track 2 = Val, Track 3 = Test)
    tracks = [1] * 24 + [2] * 8 + [3] * 8
    # Assign canonical behaviour classes
    behaviours = []
    classes = TARGET_CLASSES
    for i in range(n_samples):
        behaviours.append(classes[i % len(classes)])

    metadata = pd.DataFrame(
        {
            "sequence_id": list(range(n_samples)),
            "track_id": tracks,
            "start_timestamp_seconds": [i * 0.5 for i in range(n_samples)],
            "end_timestamp_seconds": [i * 0.5 + 1.7 for i in range(n_samples)],
            "dominant_behaviour": behaviours,
        }
    )

    split = prepare_track_grouped_splits(seqs, metadata, random_seed=42)
    return split, metadata


# 1. Baseline Model Tests
def test_frame_cnn_classifier_shapes():
    """Verify FrameCNNClassifier forward pass for 2D and 3D inputs."""
    model = FrameCNNClassifier(input_size=CNN_FEATURE_DIM, hidden_size=64, num_classes=6)

    # 3D Sequence input (batch_size=4, seq_len=10, feat_dim=512)
    x3d = torch.randn(4, 10, CNN_FEATURE_DIM)
    out3d = model(x3d)
    assert out3d.shape == (4, 6)

    # 2D Frame input (batch_size=4, feat_dim=512)
    x2d = torch.randn(4, CNN_FEATURE_DIM)
    out2d = model(x2d)
    assert out2d.shape == (4, 6)

    cfg = model.get_config()
    assert cfg["model_type"] == MODEL_FRAME_CNN
    assert "parameters" in cfg


def test_train_frame_cnn_baseline(mock_dataset_split):
    """Verify training of FrameCNNClassifier on split."""
    split, _ = mock_dataset_split
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt"
        res_dir = Path(tmpdir) / "res"

        cfg = TrainConfig(model_type=MODEL_FRAME_CNN, epochs=2, batch_size=4)
        model, train_res = train_frame_cnn_baseline(
            config=cfg,
            split=split,
            checkpoint_dir=ckpt_dir,
            results_dir=res_dir,
        )

        assert isinstance(model, FrameCNNClassifier)
        assert train_res.best_val_loss >= 0.0
        assert (ckpt_dir / f"{MODEL_FRAME_CNN}_best.pt").exists()
        assert (res_dir / "history.csv").exists()


def test_run_baseline_comparison(mock_dataset_split):
    """Verify 4-way baseline comparison across Frame-level CNN, RNN, LSTM, and GRU."""
    split, _ = mock_dataset_split

    frame_cnn = FrameCNNClassifier()
    rnn = RNNClassifier(hidden_size=32)
    lstm = LSTMClassifier(hidden_size=32)
    gru = GRUClassifier(hidden_size=32)

    models_dict = {
        MODEL_FRAME_CNN: frame_cnn,
        MODEL_RNN: rnn,
        MODEL_LSTM: lstm,
        MODEL_GRU: gru,
    }

    _, _, test_loader = create_split_dataloaders(split=split, batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        res_dir = Path(tmpdir)
        summary = run_baseline_comparison(
            models=models_dict,
            test_loader=test_loader,
            results_dir=res_dir,
        )

        assert isinstance(summary, BaselineComparisonSummary)
        df = summary.comparison_df
        assert len(df) == 4
        assert set(df.columns) >= {"Model", "Accuracy", "Precision", "Recall", "F1-score", "Weighted F1"}

        # Verify no NaN values
        assert not df.isnull().values.any()
        assert summary.best_model_name in ["Frame-level CNN", "CNN + RNN", "CNN + LSTM", "CNN + GRU"]


# 2. Ablation Tests
def test_architecture_ablation():
    """Verify architecture ablation trial generation and metric computation."""
    cm = np.eye(6, dtype=int) * 5
    m_base = EvaluationMetrics(
        model_type=MODEL_FRAME_CNN,
        accuracy=0.60,
        macro_f1=0.55,
        weighted_f1=0.55,
        macro_precision=0.58,
        macro_recall=0.55,
        per_class_metrics={},
        confusion_matrix=cm,
        y_true=[0, 1],
        y_pred=[0, 1],
        y_conf=[0.8, 0.9],
    )
    m_lstm = EvaluationMetrics(
        model_type=MODEL_LSTM,
        accuracy=0.75,
        macro_f1=0.70,
        weighted_f1=0.70,
        macro_precision=0.72,
        macro_recall=0.70,
        per_class_metrics={},
        confusion_matrix=cm,
        y_true=[0, 1],
        y_pred=[0, 1],
        y_conf=[0.9, 0.95],
    )

    trial = run_architecture_ablation(m_base, m_lstm)
    assert isinstance(trial, AblationTrial)
    assert trial.macro_f1_delta == pytest.approx(0.15, abs=1e-3)
    assert trial.pct_change > 0.0
    assert "recurrent" in trial.scientific_rationale.lower()


def test_loss_weighting_ablation(mock_dataset_split):
    """Verify class-weighted vs unweighted cross-entropy loss ablation."""
    split, _ = mock_dataset_split
    trial = run_loss_weighting_ablation(split, model_type=MODEL_LSTM, epochs=2)
    assert isinstance(trial, AblationTrial)
    assert trial.category == "Loss Formulation"
    assert trial.baseline_accuracy >= 0.0
    assert trial.experimental_accuracy >= 0.0


# 3. Temporal Error Analysis Tests
def test_transition_distance_computation():
    """Verify distance computation to behaviour transitions."""
    df = pd.DataFrame(
        {
            "sequence_id": [0, 1, 2, 3],
            "track_id": [1, 1, 1, 1],
            "start_timestamp_seconds": [0.0, 1.0, 2.0, 3.0],
            "dominant_behaviour": ["Looking toward the instructional activity", "Looking toward the instructional activity", "Looking away", "Looking away"],
        }
    )
    dists = compute_track_transition_distances(df)
    assert len(dists) == 4
    # Transition occurs between 1.0s and 2.0s -> midpoint = 1.5s
    # Distances from 1.5s: 0.0 -> 1.5s, 1.0 -> 0.5s, 2.0 -> 0.5s, 3.0 -> 1.5s
    assert dists[1] == pytest.approx(0.5, abs=1e-3)
    assert dists[2] == pytest.approx(0.5, abs=1e-3)


def test_duration_tier_and_visual_similarity():
    """Verify duration tier categorization and similar pair matching."""
    assert categorize_duration_tier(1.5) == "Fleeting (< 2.0s)"
    assert categorize_duration_tier(3.0) == "Moderate (2.0s – 5.0s)"
    assert categorize_duration_tier(6.0) == "Sustained (>= 5.0s)"

    assert is_visually_similar(CLASS_LOOKING_TOWARD_INSTRUCTION, CLASS_LOOKING_AWAY)
    assert is_visually_similar(CLASS_READING_WRITING, CLASS_MOBILE_DEVICE_ACTIVITY)
    assert not is_visually_similar(CLASS_LOOKING_TOWARD_INSTRUCTION, CLASS_HEAD_DOWN)


def test_analyze_temporal_errors():
    """Verify temporal error analyzer aggregates and case studies."""
    y_true = [0, 0, 1, 1, 2, 3]
    y_pred = [0, 1, 1, 0, 2, 3]  # Errors at index 1 and index 3
    y_conf = [0.9, 0.6, 0.85, 0.55, 0.95, 0.8]
    cm = np.zeros((6, 6), dtype=int)

    metrics = EvaluationMetrics(
        model_type=MODEL_LSTM,
        accuracy=4 / 6,
        macro_f1=0.65,
        weighted_f1=0.65,
        macro_precision=0.65,
        macro_recall=0.65,
        per_class_metrics={},
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred,
        y_conf=y_conf,
    )

    test_meta = pd.DataFrame(
        {
            "sequence_id": list(range(6)),
            "track_id": [1, 1, 1, 2, 2, 2],
            "start_timestamp_seconds": [0.0, 0.5, 1.0, 0.0, 1.0, 2.0],
            "end_timestamp_seconds": [1.7, 2.2, 2.7, 1.7, 2.7, 3.7],
            "dominant_behaviour": [TARGET_CLASSES[y] for y in y_true],
        }
    )

    segments = [
        TeachingActivitySegment(
            segment_id=1,
            video_id="test_video",
            activity_class="Lecture",
            start_timestamp_seconds=0.0,
            end_timestamp_seconds=5.0,
            duration_seconds=5.0,
            annotation_source="manual",
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = analyze_temporal_errors(
            eval_metrics=metrics,
            test_metadata_df=test_meta,
            activity_segments=segments,
            results_dir=Path(tmpdir),
        )

        assert isinstance(summary, TemporalErrorAnalysisSummary)
        assert summary.total_test_samples == 6
        assert summary.total_errors == 2
        assert summary.overall_error_rate == pytest.approx(2 / 6, abs=1e-3)
        assert not summary.detailed_records_df.empty
        assert len(summary.misclassification_case_studies) == 2


# 4. Reporting & Limitations Tests
def test_per_class_table_and_limitations():
    """Verify per-class table formatting and missing/low representation detection."""
    cm = np.zeros((6, 6), dtype=int)
    per_class = {
        CLASS_LOOKING_TOWARD_INSTRUCTION: {"precision": 0.85, "recall": 0.80, "f1": 0.82, "support": 15},
        CLASS_READING_WRITING: {"precision": 0.70, "recall": 0.65, "f1": 0.67, "support": 4},  # Low (<5)
        CLASS_MOBILE_DEVICE_ACTIVITY: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},  # Missing (0)
    }

    metrics = EvaluationMetrics(
        model_type=MODEL_LSTM,
        accuracy=0.75,
        macro_f1=0.50,
        weighted_f1=0.60,
        macro_precision=0.52,
        macro_recall=0.48,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        y_true=[0],
        y_pred=[0],
        y_conf=[0.9],
    )

    df, limitations = generate_per_class_table(metrics)
    assert len(df) == 6  # All 6 canonical classes
    assert len(limitations) >= 2  # Low representation and missing detected

    # Check status tags
    row_missing = df[df["Observable Behaviour Class"] == CLASS_MOBILE_DEVICE_ACTIVITY]
    assert "Missing" in row_missing["Data Representation Status"].values[0]

    row_low = df[df["Observable Behaviour Class"] == CLASS_READING_WRITING]
    assert "Low Representation" in row_low["Data Representation Status"].values[0]


def test_identify_top_confused_classes():
    """Verify identification of top off-diagonal misclassifications."""
    cm = np.zeros((6, 6), dtype=int)
    cm[0, 0] = 20
    cm[0, 3] = 5  # 5 samples of Looking toward instruction misclassified as Looking away
    cm[1, 4] = 3  # 3 samples of Reading/writing misclassified as Mobile-device activity

    top_conf = identify_top_confused_classes(cm, top_k=2)
    assert len(top_conf) == 2
    assert top_conf.iloc[0]["True Behaviour"] == CLASS_LOOKING_TOWARD_INSTRUCTION
    assert top_conf.iloc[0]["Predicted Behaviour"] == CLASS_LOOKING_AWAY
    assert top_conf.iloc[0]["Misclassified Count"] == 5


def test_synthesize_research_conclusions_no_mental_claims():
    """Verify conclusions strictly exclude forbidden mental state / attention terms."""
    comp_df = pd.DataFrame(
        [
            {"Model": "Frame-level CNN", "F1-score": "0.5500", "Accuracy": "0.6000"},
            {"Model": "CNN + RNN", "F1-score": "0.6200", "Accuracy": "0.6500"},
            {"Model": "CNN + LSTM", "F1-score": "0.7200", "Accuracy": "0.7600"},
            {"Model": "CNN + GRU", "F1-score": "0.7000", "Accuracy": "0.7400"},
        ]
    )
    b_summary = BaselineComparisonSummary(
        comparison_df=comp_df,
        metrics_by_model={},
        inference_latencies_ms={},
        parameter_counts={},
        best_model_name="CNN + LSTM",
        best_macro_f1=0.7200,
        recurrence_gain_pct=30.9,
    )

    err_summary = TemporalErrorAnalysisSummary(
        total_test_samples=20,
        total_errors=4,
        overall_error_rate=0.20,
        boundary_error_rate=0.40,
        steady_state_error_rate=0.10,
        boundary_error_multiplier=4.0,
        fleeting_error_rate=0.50,
        moderate_error_rate=0.20,
        sustained_error_rate=0.08,
        similar_pairs_error_count=2,
        similar_pairs_share_of_errors=0.50,
        activity_error_rates={"Lecture": 0.15},
        detailed_records_df=pd.DataFrame(),
        misclassification_case_studies=pd.DataFrame(),
    )

    conclusions = synthesize_research_conclusions(b_summary, err_summary)
    assert len(conclusions) >= 4

    # Academic compliance assertions: zero forbidden terms
    full_text = " ".join(conclusions).lower()
    for bad_term in FORBIDDEN_TERMS:
        assert bad_term not in full_text, f"Forbidden mental term '{bad_term}' found in conclusions!"

    assert "f1-score of 0.7200" in full_text
    assert "frame-level cnn" in full_text


# 5. Visualization Tests
def test_visualizations_create_figures():
    """Verify all visualization routines generate valid matplotlib Figures."""
    # 1. Model comparison
    comp_df = pd.DataFrame(
        [
            {"Model": "Frame-level CNN", "Accuracy": 0.60, "Precision": 0.58, "Recall": 0.55, "F1-score": 0.55},
            {"Model": "CNN + LSTM", "Accuracy": 0.75, "Precision": 0.72, "Recall": 0.70, "F1-score": 0.70},
        ]
    )
    fig1 = plot_model_comparison(comp_df)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # 2. Confusion matrices
    cm = np.eye(6, dtype=int) * 10
    dummy_m = EvaluationMetrics(
        model_type=MODEL_LSTM,
        accuracy=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        per_class_metrics={},
        confusion_matrix=cm,
        y_true=[0],
        y_pred=[0],
        y_conf=[1.0],
    )
    fig2 = plot_confusion_matrices({MODEL_LSTM: dummy_m})
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    # 3. Training curves
    hist_df = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.5, 1.2, 0.9],
            "val_loss": [1.6, 1.3, 1.0],
            "train_acc": [0.4, 0.6, 0.7],
            "val_acc": [0.35, 0.55, 0.65],
        }
    )
    fig3 = plot_training_curves({MODEL_LSTM: hist_df})
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    # 4. Per-class F1
    fig4 = plot_per_class_f1({MODEL_LSTM: dummy_m})
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)

    # 5. Temporal error dynamics
    err_summary = TemporalErrorAnalysisSummary(
        total_test_samples=10,
        total_errors=2,
        overall_error_rate=0.20,
        boundary_error_rate=0.33,
        steady_state_error_rate=0.10,
        boundary_error_multiplier=3.3,
        fleeting_error_rate=0.50,
        moderate_error_rate=0.20,
        sustained_error_rate=0.10,
        similar_pairs_error_count=1,
        similar_pairs_share_of_errors=0.50,
        activity_error_rates={"Lecture": 0.20},
        detailed_records_df=pd.DataFrame(),
        misclassification_case_studies=pd.DataFrame(),
    )
    fig5 = plot_temporal_error_dynamics(err_summary)
    assert isinstance(fig5, plt.Figure)
    plt.close(fig5)

    # 6. Ablation summary
    ablation_df = pd.DataFrame(
        [
            {
                "experimental_condition": "CNN + LSTM (Recurrent)",
                "macro_f1_delta": 0.15,
                "pct_change": 27.2,
            }
        ]
    )
    fig6 = plot_ablation_summary(ablation_df)
    assert isinstance(fig6, plt.Figure)
    plt.close(fig6)
