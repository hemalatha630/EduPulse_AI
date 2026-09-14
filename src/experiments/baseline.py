"""Baseline Comparison Module for Feature 11 Research Experiments.

Implements:
1. Static Frame-Level CNN Classifier (FrameCNNClassifier) baseline operating on 512-dim
   visual feature representations without temporal recurrence.
2. Fair training routine for the Frame-Level CNN baseline using identical optimizer,
   learning rate, training-only class weights, and early stopping.
3. 4-way baseline comparison engine comparing Frame-Level CNN, CNN+RNN, CNN+LSTM,
   and CNN+GRU on the identical unseen test split.
4. Actual measured values calculation: Accuracy, Macro/Weighted Precision, Recall,
   F1-score, and inference latency.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BASELINE_COMPARISON_CSV_FILENAME,
    CNN_FEATURE_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    MODEL_FRAME_CNN,
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
    MODELS_TEMPORAL_DIR,
    RESULTS_EXPERIMENTS_DIR,
    RESULTS_TEMPORAL_DIR,
)
from src.temporal.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_TARGET_CLASSES,
    SupervisedSequenceDataset,
    TARGET_CLASSES,
    TemporalDatasetSplit,
    create_split_dataloaders,
)
from src.temporal.evaluator import EvaluationMetrics, evaluate_temporal_model
from src.temporal.models import BaseTemporalClassifier
from src.temporal.trainer import TrainConfig, TrainResult, set_seed


class FrameCNNClassifier(nn.Module):
    """Static Frame-Level CNN Observable Behaviour Classifier.

    Acts as the non-recurrent baseline. Evaluates single-frame visual embeddings (512-dim)
    without any temporal sequence recurrence. When given a temporal sequence tensor
    of shape (batch_size, sequence_length, feature_dimension), it indexes the final
    time-step (t = -1) to predict the behaviour label for that instant, providing
    a strictly fair baseline comparison against recurrent models that ingest the
    entire sequence window x_{1:L} to predict the label at time t_L.
    """

    def __init__(
        self,
        input_size: int = CNN_FEATURE_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int = NUM_TARGET_CLASSES,
    ):
        super().__init__()
        self.model_type = MODEL_FRAME_CNN
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frame-level classification head.

        Args:
            x: Tensor of shape (batch_size, sequence_length, feature_dimension)
               or (batch_size, feature_dimension).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        if x.ndim == 3:
            # Extract final time-step embedding (t_L)
            frame_features = x[:, -1, :]
        elif x.ndim == 2:
            frame_features = x
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        logits = self.classifier(frame_features)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_config(self) -> dict:
        """Return model hyperparameter specification dictionary."""
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout_rate,
            "num_classes": self.num_classes,
            "parameters": self.count_parameters(),
        }


def train_frame_cnn_baseline(
    config: Optional[TrainConfig] = None,
    split: Optional[TemporalDatasetSplit] = None,
    checkpoint_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    epoch_callback: Optional[Callable[[int, int, float, float, float, float], None]] = None,
) -> Tuple[FrameCNNClassifier, TrainResult]:
    """Train the Frame-Level CNN baseline under fair, identical conditions.

    Args:
        config: Training hyperparameter config (defaults to standard TrainConfig).
        split: TemporalDatasetSplit holding train/val splits.
        checkpoint_dir: Path to save frame_cnn_best.pt.
        results_dir: Path to save training history.
        epoch_callback: Optional callback for UI progress reporting.

    Returns:
        Tuple of (trained_model, train_result).
    """
    cfg = config or TrainConfig(model_type=MODEL_FRAME_CNN)
    set_seed(cfg.seed)
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if device.type == "cuda" else "CPU"

    ckpt_dir = checkpoint_dir or MODELS_TEMPORAL_DIR
    res_dir = results_dir or (RESULTS_EXPERIMENTS_DIR / MODEL_FRAME_CNN)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    model = FrameCNNClassifier(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
    )
    model.to(device)

    if split is None:
        raise ValueError("split argument must be provided to train Frame-Level CNN baseline.")

    train_loader, val_loader, _ = create_split_dataloaders(
        split=split,
        batch_size=cfg.batch_size,
    )

    if cfg.use_class_weights and split.class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=split.class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history_records: List[dict] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
    best_state_dict = None

    checkpoint_file = ckpt_dir / f"{MODEL_FRAME_CNN}_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        train_loss = running_train_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                running_val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += len(batch_y)

        val_loss = running_val_loss / max(1, total_val)
        val_acc = correct_val / max(1, total_val)

        history_records.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )

        if epoch_callback:
            epoch_callback(epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            torch.save(
                {
                    "model_type": MODEL_FRAME_CNN,
                    "model_state_dict": best_state_dict,
                    "config": cfg.to_dict(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "label_mapping": CLASS_TO_IDX,
                    "idx_to_class": IDX_TO_CLASS,
                    "device_trained_on": device_name,
                },
                checkpoint_file,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                stopped_early = True
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(res_dir / "history.csv", index=False)

    duration = time.time() - start_time
    train_res = TrainResult(
        model_type=MODEL_FRAME_CNN,
        history_df=history_df,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        checkpoint_path=checkpoint_file,
        training_time_seconds=duration,
        stopped_early=stopped_early,
        device_used=device_name,
        config=cfg,
    )

    return model, train_res


@dataclass
class BaselineComparisonSummary:
    """Consolidated 4-way baseline comparison results."""

    comparison_df: pd.DataFrame
    metrics_by_model: Dict[str, EvaluationMetrics]
    inference_latencies_ms: Dict[str, float]
    parameter_counts: Dict[str, int]
    best_model_name: str
    best_macro_f1: float
    recurrence_gain_pct: float  # Percentage gain of best recurrent model over frame-level CNN


def run_baseline_comparison(
    models: Dict[str, Union[BaseTemporalClassifier, FrameCNNClassifier]],
    test_loader: DataLoader,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> BaselineComparisonSummary:
    """Execute rigorous 4-way baseline comparison on the identical unseen test split.

    Compares:
    - Frame-level CNN (Static baseline)
    - CNN + RNN
    - CNN + LSTM
    - CNN + GRU

    Generates a results table containing actual measured values:
    Model, Accuracy, Precision, Recall, Macro F1, Weighted F1, and Latency.
    Guarantees no invented numbers.

    Args:
        models: Dictionary mapping model key to instantiated trained model.
        test_loader: DataLoader yielding test sequences.
        results_dir: Directory to save baseline_comparison.csv.
        device: PyTorch device.

    Returns:
        BaselineComparisonSummary object.
    """
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = results_dir or RESULTS_EXPERIMENTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_map: Dict[str, EvaluationMetrics] = {}
    latencies: Dict[str, float] = {}
    param_counts: Dict[str, int] = {}
    table_rows: List[dict] = []

    # Display names
    display_names = {
        MODEL_FRAME_CNN: "Frame-level CNN",
        MODEL_RNN: "CNN + RNN",
        MODEL_LSTM: "CNN + LSTM",
        MODEL_GRU: "CNN + GRU",
    }

    for key, model in models.items():
        model.eval()
        model.to(dev)

        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[key] = params

        # Measure inference latency on test set
        start_t = time.perf_counter()
        total_samples = 0
        with torch.no_grad():
            for bx, _, _ in test_loader:
                bx = bx.to(dev)
                _ = model(bx)
                total_samples += len(bx)
        total_time_ms = (time.perf_counter() - start_t) * 1000.0
        latency_ms = total_time_ms / max(1, total_samples)
        latencies[key] = round(latency_ms, 3)

        # Evaluate test metrics
        # evaluate_temporal_model works for any module returning logits
        metrics = evaluate_temporal_model(
            model=model,  # type: ignore
            test_loader=test_loader,
            results_dir=out_dir / key,
            device=dev,
        )
        metrics_map[key] = metrics

        name = display_names.get(key.lower(), key.upper())
        table_rows.append(
            {
                "Model": name,
                "Accuracy": round(float(metrics.accuracy), 4),
                "Precision": round(float(metrics.macro_precision), 4),
                "Recall": round(float(metrics.macro_recall), 4),
                "F1-score": round(float(metrics.macro_f1), 4),
                "Weighted F1": round(float(metrics.weighted_f1), 4),
                "Parameters": params,
                "Latency (ms/sample)": latencies[key],
            }
        )

    df = pd.DataFrame(table_rows)
    # Save CSV
    df.to_csv(out_dir / BASELINE_COMPARISON_CSV_FILENAME, index=False)

    # Calculate best model and recurrence gain
    recurrent_keys = [k for k in models.keys() if k != MODEL_FRAME_CNN]
    best_recurrent_key = max(recurrent_keys, key=lambda k: metrics_map[k].macro_f1) if recurrent_keys else MODEL_FRAME_CNN
    best_model_name = display_names.get(best_recurrent_key, best_recurrent_key)
    best_macro_f1 = float(metrics_map[best_recurrent_key].macro_f1) if best_recurrent_key in metrics_map else 0.0

    frame_cnn_f1 = float(metrics_map[MODEL_FRAME_CNN].macro_f1) if MODEL_FRAME_CNN in metrics_map else 0.0
    if frame_cnn_f1 > 0:
        gain_pct = ((best_macro_f1 - frame_cnn_f1) / frame_cnn_f1) * 100.0
    else:
        gain_pct = 0.0

    summary = BaselineComparisonSummary(
        comparison_df=df,
        metrics_by_model=metrics_map,
        inference_latencies_ms=latencies,
        parameter_counts=param_counts,
        best_model_name=best_model_name,
        best_macro_f1=best_macro_f1,
        recurrence_gain_pct=round(gain_pct, 2),
    )

    return summary
