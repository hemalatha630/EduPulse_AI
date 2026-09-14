"""Training Engine and Early Stopping Module for Feature 8 Temporal Models.

Implements fair, reproducible training for Vanilla RNN, LSTM, and GRU models:
- Identical training loop, optimizer, and loss formulation
- Seed setting across Python, NumPy, and PyTorch
- Early stopping based strictly on validation loss (test set is never touched)
- Best checkpoint saving to models/temporal/<model_type>_best.pt
- Live epoch progress reporting callback for Streamlit UI
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    CNN_FEATURE_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
    MODELS_TEMPORAL_DIR,
    RESULTS_TEMPORAL_DIR,
    SUPPORTED_TEMPORAL_MODELS,
)
from src.temporal.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_TARGET_CLASSES,
    TemporalDatasetSplit,
    create_split_dataloaders,
)
from src.temporal.models import BaseTemporalClassifier, create_temporal_model


@dataclass
class TrainConfig:
    """Hyperparameter and runtime configuration for temporal model training."""

    model_type: str = MODEL_LSTM
    input_size: int = CNN_FEATURE_DIM
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    num_classes: int = NUM_TARGET_CLASSES
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    seed: int = DEFAULT_RANDOM_SEED
    use_class_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    """Summary of training execution, history, and checkpoint."""

    model_type: str
    history_df: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    checkpoint_path: Path
    training_time_seconds: float
    stopped_early: bool
    device_used: str
    config: TrainConfig

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "best_epoch": self.best_epoch,
            "best_val_loss": round(float(self.best_val_loss), 4),
            "best_val_acc": round(float(self.best_val_acc), 4),
            "checkpoint_path": str(self.checkpoint_path),
            "training_time_seconds": round(float(self.training_time_seconds), 3),
            "stopped_early": self.stopped_early,
            "device_used": self.device_used,
            "config": self.config.to_dict(),
        }


def set_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Set random seed across Python, NumPy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_temporal_model(
    config: TrainConfig,
    split: TemporalDatasetSplit,
    checkpoint_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    epoch_callback: Optional[Callable[[int, int, float, float, float, float], None]] = None,
) -> Tuple[BaseTemporalClassifier, TrainResult]:
    """Train a single temporal model (RNN, LSTM, or GRU) on the training split.

    Evaluates on validation split each epoch and saves the best model checkpoint.
    Does NOT touch the test set (maintains scientific validity).

    Args:
        config: Hyperparameter configuration.
        split: TemporalDatasetSplit object holding datasets.
        checkpoint_dir: Directory to save checkpoint (.pt).
        results_dir: Directory to save history (.csv).
        epoch_callback: Optional callback(epoch, total_epochs, tr_loss, tr_acc, val_loss, val_acc).

    Returns:
        Tuple of (trained_model_with_best_weights, train_result).
    """
    set_seed(config.seed)
    start_time = time.time()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if device.type == "cuda" else "CPU"

    # Setup directories
    ckpt_dir = checkpoint_dir or MODELS_TEMPORAL_DIR
    res_dir = results_dir or (RESULTS_TEMPORAL_DIR / config.model_type.lower())
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = create_temporal_model(
        model_type=config.model_type,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )
    model.to(device)

    # Dataloaders
    train_loader, val_loader, _ = create_split_dataloaders(
        split=split,
        batch_size=config.batch_size,
    )

    # Loss with training-only class weights
    if config.use_class_weights and split.class_weights is not None:
        weights = split.class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    history_records: List[dict] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
    best_state_dict = None

    checkpoint_file = ckpt_dir / f"{config.model_type.lower()}_best.pt"

    for epoch in range(1, config.epochs + 1):
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

            # Gradient clipping to stabilize recurrent gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            epoch_callback(epoch, config.epochs, train_loss, train_acc, val_loss, val_acc)

        # Check for improvement (early stopping)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Save best checkpoint
            torch.save(
                {
                    "model_type": config.model_type.lower(),
                    "model_state_dict": best_state_dict,
                    "config": config.to_dict(),
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
            if patience_counter >= config.patience:
                stopped_early = True
                break

    # Restore model to best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)

    history_df = pd.DataFrame(history_records)
    history_csv = res_dir / "history.csv"
    history_df.to_csv(history_csv, index=False)

    training_duration = time.time() - start_time

    result = TrainResult(
        model_type=config.model_type.lower(),
        history_df=history_df,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        checkpoint_path=checkpoint_file,
        training_time_seconds=training_duration,
        stopped_early=stopped_early,
        device_used=device_name,
        config=config,
    )

    return model, result


def train_all_temporal_models(
    base_config: TrainConfig,
    split: TemporalDatasetSplit,
    checkpoint_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    model_callback: Optional[Callable[[str, int, int], None]] = None,
    epoch_callback: Optional[Callable[[str, int, int, float, float, float, float], None]] = None,
) -> Dict[str, Tuple[BaseTemporalClassifier, TrainResult]]:
    """Train Vanilla RNN, LSTM, and GRU sequentially under identical experimental conditions.

    Guarantees strict fair comparison: same dataset, same split, same seed,
    same hyperparameters, and same optimizer/loss.

    Args:
        base_config: Common hyperparameter configuration.
        split: TemporalDatasetSplit containing train/val data.
        checkpoint_dir: Checkpoint directory.
        results_dir: Results directory.
        model_callback: Callback when switching models: (model_type, model_idx, total_models).
        epoch_callback: Callback on each epoch: (model_type, epoch, total_epochs, tr_loss, tr_acc, val_loss, val_acc).

    Returns:
        Dict mapping model_type to (trained_model, train_result).
    """
    trained_models: Dict[str, Tuple[BaseTemporalClassifier, TrainResult]] = {}
    models_to_train = [MODEL_RNN, MODEL_LSTM, MODEL_GRU]

    for m_idx, m_type in enumerate(models_to_train, 1):
        if model_callback:
            model_callback(m_type, m_idx, len(models_to_train))

        # Copy config with specific model_type
        m_config = TrainConfig(
            model_type=m_type,
            input_size=base_config.input_size,
            hidden_size=base_config.hidden_size,
            num_layers=base_config.num_layers,
            dropout=base_config.dropout,
            num_classes=base_config.num_classes,
            learning_rate=base_config.learning_rate,
            batch_size=base_config.batch_size,
            epochs=base_config.epochs,
            patience=base_config.patience,
            seed=base_config.seed,
            use_class_weights=base_config.use_class_weights,
        )

        def make_epoch_cb(m_name: str):
            if not epoch_callback:
                return None
            return lambda ep, total_ep, tr_l, tr_a, val_l, val_a: epoch_callback(
                m_name, ep, total_ep, tr_l, tr_a, val_l, val_a
            )

        model, res = train_temporal_model(
            config=m_config,
            split=split,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            epoch_callback=make_epoch_cb(m_type),
        )
        trained_models[m_type] = (model, res)

    return trained_models
