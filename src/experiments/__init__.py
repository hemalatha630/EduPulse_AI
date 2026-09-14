"""Research Experiments, Ablation, and Temporal Error Analysis Module (Feature 11).

Provides:
- FrameCNNClassifier and 4-way baseline comparison engine
- Controlled single-variable ablation studies
- Temporal error analysis across transitions, duration tiers, and classroom activities
- Evidence-based conclusion and data limitation reporting
- Scientific visualization figures
"""

from src.experiments.ablation import (
    AblationTrial,
    run_all_ablation_studies,
    run_architecture_ablation,
    run_loss_weighting_ablation,
    run_sequence_length_ablation,
    run_stride_ablation,
)
from src.experiments.baseline import (
    BaselineComparisonSummary,
    FrameCNNClassifier,
    run_baseline_comparison,
    train_frame_cnn_baseline,
)
from src.experiments.error_analysis import (
    SIMILAR_PAIRS,
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
    SHORT_CLASS_NAMES,
    plot_ablation_summary,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_per_class_f1,
    plot_temporal_error_dynamics,
    plot_training_curves,
)

__all__ = [
    # Baseline
    "FrameCNNClassifier",
    "train_frame_cnn_baseline",
    "run_baseline_comparison",
    "BaselineComparisonSummary",
    # Ablation
    "AblationTrial",
    "run_architecture_ablation",
    "run_sequence_length_ablation",
    "run_stride_ablation",
    "run_loss_weighting_ablation",
    "run_all_ablation_studies",
    # Error analysis
    "TemporalErrorRecord",
    "TemporalErrorAnalysisSummary",
    "analyze_temporal_errors",
    "compute_track_transition_distances",
    "categorize_duration_tier",
    "is_visually_similar",
    "SIMILAR_PAIRS",
    # Reporting
    "generate_per_class_table",
    "identify_top_confused_classes",
    "synthesize_research_conclusions",
    "FORBIDDEN_TERMS",
    # Visualization
    "plot_model_comparison",
    "plot_confusion_matrices",
    "plot_training_curves",
    "plot_per_class_f1",
    "plot_temporal_error_dynamics",
    "plot_ablation_summary",
    "SHORT_CLASS_NAMES",
]
