"""Temporal sequence creation and recurrent modelling package for EduPulse AI.

Provides:
- Sliding-window sequence creation & serialization (Feature 7)
- Track-grouped data leakage prevention & PyTorch Datasets (Feature 8)
- Recurrent neural models: Vanilla RNN, LSTM, GRU (Feature 8)
- Reproducible training loop, class weighting, early stopping, and checkpointing (Feature 8)
- Comprehensive test evaluation, model comparison, and single-sequence inference (Feature 8)
- Matplotlib visualizations for timelines, coverage, training curves, and confusion matrices
"""

from src.config import (
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
    SUPPORTED_TEMPORAL_MODELS,
)
from src.temporal.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_TARGET_CLASSES,
    SupervisedSequenceDataset,
    TARGET_CLASSES,
    TemporalDatasetSplit,
    compute_training_class_weights,
    create_split_dataloaders,
    encode_behaviour_labels,
    prepare_track_grouped_splits,
)
from src.temporal.evaluator import (
    EvaluationMetrics,
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
from src.temporal.sequence_generator import (
    ClassroomSequenceDataset,
    SequenceSummary,
    TemporalSequenceGenerator,
    run_temporal_sequence_creation,
    sequences_to_tensor,
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
    create_sequence_timeline_figure,
    create_track_coverage_figure,
    create_training_curves_figure,
)

__all__ = [
    # Feature 7
    "TemporalSequenceGenerator",
    "SequenceSummary",
    "run_temporal_sequence_creation",
    "sequences_to_tensor",
    "ClassroomSequenceDataset",
    "create_sequence_timeline_figure",
    "create_track_coverage_figure",
    # Feature 8 - Dataset & Splitting
    "TARGET_CLASSES",
    "NUM_TARGET_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "SupervisedSequenceDataset",
    "TemporalDatasetSplit",
    "encode_behaviour_labels",
    "compute_training_class_weights",
    "prepare_track_grouped_splits",
    "create_split_dataloaders",
    # Feature 8 - Models
    "MODEL_RNN",
    "MODEL_LSTM",
    "MODEL_GRU",
    "SUPPORTED_TEMPORAL_MODELS",
    "BaseTemporalClassifier",
    "RNNClassifier",
    "LSTMClassifier",
    "GRUClassifier",
    "create_temporal_model",
    # Feature 8 - Training
    "TrainConfig",
    "TrainResult",
    "set_seed",
    "train_temporal_model",
    "train_all_temporal_models",
    # Feature 8 - Evaluation & Inference
    "EvaluationMetrics",
    "evaluate_temporal_model",
    "compare_temporal_models",
    "predict_sequence",
    "generate_batch_predictions",
    # Feature 8 - Visualizations
    "create_training_curves_figure",
    "create_confusion_matrix_figure",
    "create_model_comparison_figure",
]
