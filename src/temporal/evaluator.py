"""Evaluation, Fair Comparison, and Inference Module for Feature 8 Temporal Models.

Calculates test set metrics strictly on unseen data:
- Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall
- Per-class precision, recall, and F1 across the 6 observable behaviour categories
- 6x6 Confusion Matrix
- Fair comparison table comparing RNN, LSTM, and GRU side-by-side
- Single-sequence inference and batch prediction saving (predictions.csv)
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from torch.utils.data import DataLoader

from src.config import (
    PREDICTIONS_CSV_FILENAME,
    RESULTS_TEMPORAL_DIR,
)
from src.temporal.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_TARGET_CLASSES,
    SupervisedSequenceDataset,
    TARGET_CLASSES,
    TemporalDatasetSplit,
)
from src.temporal.models import BaseTemporalClassifier


@dataclass
class EvaluationMetrics:
    """Comprehensive test-set evaluation metrics for a temporal model."""

    model_type: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    y_true: List[int]
    y_pred: List[int]
    y_conf: List[float]

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "accuracy": round(float(self.accuracy), 4),
            "macro_f1": round(float(self.macro_f1), 4),
            "weighted_f1": round(float(self.weighted_f1), 4),
            "macro_precision": round(float(self.macro_precision), 4),
            "macro_recall": round(float(self.macro_recall), 4),
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def evaluate_temporal_model(
    model: BaseTemporalClassifier,
    test_loader: DataLoader,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> EvaluationMetrics:
    """Evaluate a trained temporal model on the unseen test set.

    Args:
        model: Trained recurrent classifier.
        test_loader: DataLoader yielding test sequences and targets.
        results_dir: Optional path to save metrics.json.
        device: Device to run evaluation on.

    Returns:
        EvaluationMetrics instance.
    """
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to(dev)

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_confs: List[float] = []

    with torch.no_grad():
        for batch_x, batch_y, _ in test_loader:
            batch_x = batch_x.to(dev)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_targets.extend(batch_y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    # Calculate overall metrics
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix across the 6 approved classes
    labels_order = list(range(NUM_TARGET_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    for idx, cls_name in enumerate(TARGET_CLASSES):
        # Binary mask for class idx
        cls_true = (y_true == idx)
        cls_pred = (y_pred == idx)
        support = int(np.sum(cls_true))

        if support > 0 or np.sum(cls_pred) > 0:
            p = float(precision_score(cls_true, cls_pred, zero_division=0))
            r = float(recall_score(cls_true, cls_pred, zero_division=0))
            f = float(f1_score(cls_true, cls_pred, zero_division=0))
        else:
            p, r, f = 0.0, 0.0, 0.0

        per_class[cls_name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": support,
        }

    metrics = EvaluationMetrics(
        model_type=model.model_type,
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        y_true=all_targets,
        y_pred=all_preds,
        y_conf=all_confs,
    )

    # Save metrics JSON
    out_dir = results_dir or (RESULTS_TEMPORAL_DIR / model.model_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    return metrics


def compare_temporal_models(
    metrics_dict: Dict[str, EvaluationMetrics],
) -> pd.DataFrame:
    """Generate a clean tabular comparison across trained temporal models.

    Args:
        metrics_dict: Dictionary mapping model name (e.g. 'rnn', 'lstm', 'gru') to EvaluationMetrics.

    Returns:
        DataFrame with columns: Model, Accuracy, Macro F1, Weighted F1, Precision, Recall.
    """
    rows = []
    for m_type, m_val in metrics_dict.items():
        rows.append(
            {
                "Model": m_type.upper(),
                "Accuracy": f"{m_val.accuracy:.2%}",
                "Macro F1": f"{m_val.macro_f1:.4f}",
                "Weighted F1": f"{m_val.weighted_f1:.4f}",
                "Precision": f"{m_val.macro_precision:.4f}",
                "Recall": f"{m_val.macro_recall:.4f}",
            }
        )
    df = pd.DataFrame(rows)
    return df


def predict_sequence(
    model: BaseTemporalClassifier,
    sequence: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """Perform single-sequence inference.

    Args:
        model: Trained recurrent classifier.
        sequence: Array of shape (L, D) representing one temporal sequence window.
        device: Device to run inference on.

    Returns:
        Tuple of:
        - predicted_class: String name of top predicted observable behaviour.
        - confidence: Float confidence score in [0.0, 1.0].
        - probabilities: Dict mapping each behaviour class to its probability.
    """
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to(dev)

    if sequence.ndim == 2:
        seq_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(dev)  # (1, L, D)
    elif sequence.ndim == 3:
        seq_tensor = torch.from_numpy(sequence).float().to(dev)
    else:
        raise ValueError(f"Sequence array must have 2 or 3 dimensions, got shape {sequence.shape}")

    with torch.no_grad():
        logits = model(seq_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top_class = IDX_TO_CLASS.get(top_idx, "Unknown")

    prob_dict = {
        TARGET_CLASSES[i]: round(float(probs[i]), 4)
        for i in range(len(TARGET_CLASSES))
    }

    return top_class, round(top_conf, 4), prob_dict


def generate_batch_predictions(
    model: BaseTemporalClassifier,
    dataset: SupervisedSequenceDataset,
    video_id: str,
    output_csv_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Generate and save row-by-row prediction records across a dataset.

    Args:
        model: Trained recurrent classifier.
        dataset: SupervisedSequenceDataset.
        video_id: Video identifier stem.
        output_csv_path: Path to save predictions.csv.
        device: Device for inference.

    Returns:
        DataFrame of prediction records.
    """
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to(dev)

    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    records = []

    with torch.no_grad():
        for batch_x, batch_y, batch_meta in loader:
            batch_x = batch_x.to(dev)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            confs_np = confs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            y_np = batch_y.cpu().numpy()

            batch_len = len(preds_np)
            for i in range(batch_len):
                pred_idx = int(preds_np[i])
                true_idx = int(y_np[i])

                def _extract_meta(field_key: str, default_val):
                    if field_key not in batch_meta:
                        return default_val
                    val = batch_meta[field_key]
                    if isinstance(val, (torch.Tensor, np.ndarray, list)):
                        if i < len(val):
                            v = val[i]
                            return v.item() if hasattr(v, "item") else v
                        return default_val
                    return val.item() if hasattr(val, "item") else val

                seq_id = _extract_meta("sequence_id", i)
                track_id = _extract_meta("track_id", 1)
                start_t = _extract_meta("start_timestamp_seconds", 0.0)
                end_t = _extract_meta("end_timestamp_seconds", 0.0)

                records.append(
                    {
                        "model": model.model_type.upper(),
                        "video_id": video_id,
                        "track_id": int(track_id) if hasattr(track_id, "__int__") else track_id,
                        "sequence_id": int(seq_id) if hasattr(seq_id, "__int__") else seq_id,
                        "start_timestamp_seconds": round(float(start_t), 3),
                        "end_timestamp_seconds": round(float(end_t), 3),
                        "true_label": IDX_TO_CLASS.get(true_idx, "Unknown"),
                        "predicted_label": IDX_TO_CLASS.get(pred_idx, "Unknown"),
                        "confidence": round(float(confs_np[i]), 4),
                    }
                )

    df = pd.DataFrame(records)

    out_file = output_csv_path or (RESULTS_TEMPORAL_DIR / PREDICTIONS_CSV_FILENAME)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    return df
