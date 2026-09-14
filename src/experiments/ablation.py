"""Ablation Studies Module for Feature 11 Research Experiments.

Performs controlled, single-variable empirical ablation experiments:
1. Architectural Recurrence Benefit: Static Frame-Level CNN vs Recurrent Temporal Models.
2. Sequence Window Length (L): Evaluates impact of temporal context span (e.g. L=5, 10, 15).
3. Temporal Stride (S): Evaluates impact of window step size and temporal overlap (e.g. S=1, 2, 5).
4. Loss Class-Weighting: Evaluates impact of inverse-frequency class-weighted cross-entropy vs unweighted loss.

Each trial records:
- Experiment category & manipulated variable
- Baseline condition vs Experimental condition
- What changed and scientific rationale
- Test Accuracy, Test Macro F1, and Delta F1
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    ABLATION_SUMMARY_CSV_FILENAME,
    CNN_FEATURE_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEQUENCE_STRIDE,
    MODEL_FRAME_CNN,
    MODEL_LSTM,
    MODELS_TEMPORAL_DIR,
    RESULTS_ABLATION_DIR,
    RESULTS_EXPERIMENTS_DIR,
)
from src.experiments.baseline import (
    BaselineComparisonSummary,
    FrameCNNClassifier,
    train_frame_cnn_baseline,
)
from src.temporal.dataset import (
    NUM_TARGET_CLASSES,
    SupervisedSequenceDataset,
    TemporalDatasetSplit,
    create_split_dataloaders,
    prepare_track_grouped_splits,
)
from src.temporal.evaluator import EvaluationMetrics, evaluate_temporal_model
from src.temporal.models import BaseTemporalClassifier, LSTMClassifier, create_temporal_model
from src.temporal.sequence_generator import TemporalSequenceGenerator
from src.temporal.trainer import TrainConfig, train_temporal_model


@dataclass
class AblationTrial:
    """Represents the outcome of a single controlled ablation experiment."""

    experiment_id: str
    category: str
    variable_manipulated: str
    baseline_condition: str
    experimental_condition: str
    what_changed: str
    scientific_rationale: str
    baseline_accuracy: float
    experimental_accuracy: float
    baseline_macro_f1: float
    experimental_macro_f1: float
    macro_f1_delta: float  # (experimental - baseline)
    pct_change: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_architecture_ablation(
    baseline_cnn_metrics: EvaluationMetrics,
    lstm_metrics: EvaluationMetrics,
) -> AblationTrial:
    """Ablation 1: Architectural Recurrence Benefit.

    Compares static Frame-Level CNN (0 recurrent layers) vs CNN+LSTM (baseline temporal model).
    Quantifies the precise empirical gain of sequential recurrent modelling.
    """
    base_acc = float(baseline_cnn_metrics.accuracy)
    exp_acc = float(lstm_metrics.accuracy)
    base_f1 = float(baseline_cnn_metrics.macro_f1)
    exp_f1 = float(lstm_metrics.macro_f1)

    delta = exp_f1 - base_f1
    pct = (delta / base_f1 * 100.0) if base_f1 > 0 else 0.0

    return AblationTrial(
        experiment_id="ABL-ARCH-01",
        category="Architecture",
        variable_manipulated="Temporal Recurrence (None vs LSTM)",
        baseline_condition="Frame-level CNN (static single frame)",
        experimental_condition="CNN + LSTM (10-frame recurrent window)",
        what_changed="Added 1-layer LSTM recurrent cell (hidden_size=128) over sequence of 512-dim visual embeddings.",
        scientific_rationale=(
            "Observable classroom behaviours (e.g. peer interaction, writing) have inherent temporal duration; "
            "recurrent cells allow evidence accumulation across frames rather than isolated instant decisions."
        ),
        baseline_accuracy=round(base_acc, 4),
        experimental_accuracy=round(exp_acc, 4),
        baseline_macro_f1=round(base_f1, 4),
        experimental_macro_f1=round(exp_f1, 4),
        macro_f1_delta=round(delta, 4),
        pct_change=round(pct, 2),
    )


def run_sequence_length_ablation(
    features_array: np.ndarray,
    metadata_df: pd.DataFrame,
    sequence_lengths: List[int] = [5, 10, 15],
    default_stride: int = DEFAULT_SEQUENCE_STRIDE,
    model_type: str = MODEL_LSTM,
    epochs: int = 15,
    seed: int = DEFAULT_RANDOM_SEED,
    device: Optional[torch.device] = None,
) -> List[AblationTrial]:
    """Ablation 2: Sequence Window Length (L).

    Evaluates temporal context span:
    - L=5 (~0.8s): Fleeting / local temporal context
    - L=10 (~1.7s): Standard balanced window (baseline)
    - L=15 (~2.5s): Extended temporal context
    """
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trials: List[AblationTrial] = []

    results_by_l: Dict[int, Tuple[float, float]] = {}

    for L in sequence_lengths:
        generator = TemporalSequenceGenerator(
            sequence_length=L,
            stride=default_stride,
        )
        seqs, meta, _ = generator.generate_sequences_from_features(features_array, metadata_df)

        if len(seqs) < 10:
            # Fallback if too few sequences
            continue

        split = prepare_track_grouped_splits(seqs, meta, random_seed=seed)
        cfg = TrainConfig(
            model_type=model_type,
            epochs=epochs,
            seed=seed,
        )
        model, _ = train_temporal_model(config=cfg, split=split)
        _, _, test_loader = create_split_dataloaders(split=split, batch_size=DEFAULT_BATCH_SIZE)
        m = evaluate_temporal_model(model=model, test_loader=test_loader, device=dev)
        results_by_l[L] = (float(m.accuracy), float(m.macro_f1))

    # Compare each non-baseline length against baseline L=10 (or first length)
    ref_L = 10 if 10 in results_by_l else (sequence_lengths[0] if sequence_lengths else 10)
    ref_acc, ref_f1 = results_by_l.get(ref_L, (0.0, 0.0))

    for L in sequence_lengths:
        if L not in results_by_l or L == ref_L:
            continue
        curr_acc, curr_f1 = results_by_l[L]
        delta = curr_f1 - ref_f1
        pct = (delta / ref_f1 * 100.0) if ref_f1 > 0 else 0.0

        trials.append(
            AblationTrial(
                experiment_id=f"ABL-LEN-{L:02d}",
                category="Sequence Length",
                variable_manipulated=f"Window Length L={L} vs L={ref_L}",
                baseline_condition=f"L={ref_L} frames (~{ref_L*0.17:.1f}s)",
                experimental_condition=f"L={L} frames (~{L*0.17:.1f}s)",
                what_changed=f"Changed sliding-window sequence length from {ref_L} to {L} frames while fixing stride S={default_stride}.",
                scientific_rationale=(
                    f"Shorter windows (L=5) reduce response latency but provide limited temporal context; "
                    f"longer windows (L=15) aggregate richer temporal context but risk bridging behaviour transitions."
                ),
                baseline_accuracy=round(ref_acc, 4),
                experimental_accuracy=round(curr_acc, 4),
                baseline_macro_f1=round(ref_f1, 4),
                experimental_macro_f1=round(curr_f1, 4),
                macro_f1_delta=round(delta, 4),
                pct_change=round(pct, 2),
            )
        )

    return trials


def run_stride_ablation(
    features_array: np.ndarray,
    metadata_df: pd.DataFrame,
    strides: List[int] = [1, 2, 5],
    default_length: int = DEFAULT_SEQUENCE_LENGTH,
    model_type: str = MODEL_LSTM,
    epochs: int = 15,
    seed: int = DEFAULT_RANDOM_SEED,
    device: Optional[torch.device] = None,
) -> List[AblationTrial]:
    """Ablation 3: Temporal Window Stride (S).

    Evaluates temporal stride:
    - S=1: High temporal resolution (maximum overlap)
    - S=2: Standard balanced stride (baseline)
    - S=5: Sparse temporal sampling (minimum overlap)
    """
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trials: List[AblationTrial] = []
    results_by_s: Dict[int, Tuple[float, float]] = {}

    for S in strides:
        generator = TemporalSequenceGenerator(
            sequence_length=default_length,
            stride=S,
        )
        seqs, meta, _ = generator.generate_sequences_from_features(features_array, metadata_df)

        if len(seqs) < 10:
            continue

        split = prepare_track_grouped_splits(seqs, meta, random_seed=seed)
        cfg = TrainConfig(
            model_type=model_type,
            epochs=epochs,
            seed=seed,
        )
        model, _ = train_temporal_model(config=cfg, split=split)
        _, _, test_loader = create_split_dataloaders(split=split, batch_size=DEFAULT_BATCH_SIZE)
        m = evaluate_temporal_model(model=model, test_loader=test_loader, device=dev)
        results_by_s[S] = (float(m.accuracy), float(m.macro_f1))

    ref_S = 2 if 2 in results_by_s else (strides[0] if strides else 2)
    ref_acc, ref_f1 = results_by_s.get(ref_S, (0.0, 0.0))

    for S in strides:
        if S not in results_by_s or S == ref_S:
            continue
        curr_acc, curr_f1 = results_by_s[S]
        delta = curr_f1 - ref_f1
        pct = (delta / ref_f1 * 100.0) if ref_f1 > 0 else 0.0

        trials.append(
            AblationTrial(
                experiment_id=f"ABL-STRIDE-{S:02d}",
                category="Temporal Stride",
                variable_manipulated=f"Stride S={S} vs S={ref_S}",
                baseline_condition=f"S={ref_S} frames (balanced stride)",
                experimental_condition=f"S={S} frames (modified stride)",
                what_changed=f"Changed sliding-window step stride from {ref_S} to {S} frames while fixing length L={default_length}.",
                scientific_rationale=(
                    f"Smaller stride (S=1) produces higher training sample density but increases computational cost; "
                    f"larger stride (S=5) speeds up inference but reduces temporal resolution at behaviour boundaries."
                ),
                baseline_accuracy=round(ref_acc, 4),
                experimental_accuracy=round(curr_acc, 4),
                baseline_macro_f1=round(ref_f1, 4),
                experimental_macro_f1=round(curr_f1, 4),
                macro_f1_delta=round(delta, 4),
                pct_change=round(pct, 2),
            )
        )

    return trials


def run_loss_weighting_ablation(
    split: TemporalDatasetSplit,
    model_type: str = MODEL_LSTM,
    epochs: int = 15,
    seed: int = DEFAULT_RANDOM_SEED,
    device: Optional[torch.device] = None,
) -> AblationTrial:
    """Ablation 4: Loss Class-Weighting Strategy.

    Evaluates effect of training with inverse-frequency class weights
    versus unweighted standard cross-entropy.
    """
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Baseline: With class weights (standard pipeline setting)
    cfg_weighted = TrainConfig(
        model_type=model_type,
        use_class_weights=True,
        epochs=epochs,
        seed=seed,
    )
    model_w, _ = train_temporal_model(config=cfg_weighted, split=split)
    _, _, test_loader = create_split_dataloaders(split=split, batch_size=DEFAULT_BATCH_SIZE)
    metrics_w = evaluate_temporal_model(model=model_w, test_loader=test_loader, device=dev)

    # 2. Experimental condition: Unweighted Cross-Entropy
    cfg_unweighted = TrainConfig(
        model_type=model_type,
        use_class_weights=False,
        epochs=epochs,
        seed=seed,
    )
    model_u, _ = train_temporal_model(config=cfg_unweighted, split=split)
    metrics_u = evaluate_temporal_model(model=model_u, test_loader=test_loader, device=dev)

    base_acc = float(metrics_w.accuracy)
    base_f1 = float(metrics_w.macro_f1)
    exp_acc = float(metrics_u.accuracy)
    exp_f1 = float(metrics_u.macro_f1)

    delta = exp_f1 - base_f1
    pct = (delta / base_f1 * 100.0) if base_f1 > 0 else 0.0

    return AblationTrial(
        experiment_id="ABL-LOSS-01",
        category="Loss Formulation",
        variable_manipulated="Training Class Weighting (Inverse-Frequency vs Unweighted)",
        baseline_condition="Inverse-frequency class weights enabled",
        experimental_condition="Unweighted standard cross-entropy loss",
        what_changed="Disabled inverse-frequency loss weighting during training, treating all behaviour classes equally in loss gradient computation.",
        scientific_rationale=(
            "Classroom observable behaviours suffer natural class imbalance (e.g. Looking toward instruction dominates over mobile-device activity). "
            "Without class weights, models tend to favor dominant classes, depressing minority-class recall."
        ),
        baseline_accuracy=round(base_acc, 4),
        experimental_accuracy=round(exp_acc, 4),
        baseline_macro_f1=round(base_f1, 4),
        experimental_macro_f1=round(exp_f1, 4),
        macro_f1_delta=round(delta, 4),
        pct_change=round(pct, 2),
    )


def run_all_ablation_studies(
    split: TemporalDatasetSplit,
    baseline_cnn_metrics: EvaluationMetrics,
    lstm_metrics: EvaluationMetrics,
    features_array: Optional[np.ndarray] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Consolidated runner executing all controlled ablation studies.

    Saves results to results/ablation/ablation_summary.csv.
    """
    out_dir = results_dir or RESULTS_ABLATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: List[AblationTrial] = []

    # 1. Architecture ablation
    arch_trial = run_architecture_ablation(baseline_cnn_metrics, lstm_metrics)
    trials.append(arch_trial)

    # 2. Loss weighting ablation
    loss_trial = run_loss_weighting_ablation(split, device=device)
    trials.append(loss_trial)

    # 3. Sequence length & stride ablations (if raw features provided)
    if features_array is not None and metadata_df is not None:
        try:
            len_trials = run_sequence_length_ablation(
                features_array=features_array,
                metadata_df=metadata_df,
                sequence_lengths=[5, 10, 15],
                device=device,
            )
            trials.extend(len_trials)
        except Exception as e:
            # Handle gracefully if dataset too small for multi-length
            pass

        try:
            stride_trials = run_stride_ablation(
                features_array=features_array,
                metadata_df=metadata_df,
                strides=[1, 2, 5],
                device=device,
            )
            trials.extend(stride_trials)
        except Exception as e:
            pass

    rows = [t.to_dict() for t in trials]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / ABLATION_SUMMARY_CSV_FILENAME, index=False)

    return df
