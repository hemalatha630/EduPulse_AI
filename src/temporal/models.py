"""Temporal Neural Network Architectures for EduPulse AI.

Implements three recurrent neural architectures under an identical, fair baseline:
1. Vanilla RNN (RNNClassifier)
2. Long Short-Term Memory (LSTMClassifier)
3. Gated Recurrent Unit (GRUClassifier)

All models accept input sequences of shape (batch_size, sequence_length, feature_dimension)
and map the final temporal representation to observable behaviour classification logits.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.config import (
    CNN_FEATURE_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
    SUPPORTED_TEMPORAL_MODELS,
)
from src.temporal.dataset import NUM_TARGET_CLASSES


class BaseTemporalClassifier(nn.Module):
    """Abstract base class for recurrent observable behaviour classifiers."""

    def __init__(
        self,
        model_type: str,
        input_size: int = CNN_FEATURE_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int = NUM_TARGET_CLASSES,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_classes = num_classes

        # Dropout between RNN layers (only active if num_layers > 1)
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.model_type == MODEL_RNN:
            self.recurrent = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                nonlinearity="tanh",
            )
        elif self.model_type == MODEL_LSTM:
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.model_type == MODEL_GRU:
            self.recurrent = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. Choose from {SUPPORTED_TEMPORAL_MODELS}."
            )

        # Classification Head (Dropout + Linear)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through recurrent layer and classification head.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, feature_dimension).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # Recurrent layer forward pass
        out, _ = self.recurrent(x)

        # Extract final time-step representation: (batch_size, hidden_size)
        final_rep = out[:, -1, :]

        # Regularization & projection
        dropped = self.dropout(final_rep)
        logits = self.fc(dropped)
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
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "num_classes": self.num_classes,
            "parameters": self.count_parameters(),
        }


class RNNClassifier(BaseTemporalClassifier):
    """Vanilla Recurrent Neural Network (RNN) observable behaviour classifier."""

    def __init__(
        self,
        input_size: int = CNN_FEATURE_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int = NUM_TARGET_CLASSES,
    ):
        super().__init__(
            model_type=MODEL_RNN,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )


class LSTMClassifier(BaseTemporalClassifier):
    """Long Short-Term Memory (LSTM) observable behaviour classifier."""

    def __init__(
        self,
        input_size: int = CNN_FEATURE_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int = NUM_TARGET_CLASSES,
    ):
        super().__init__(
            model_type=MODEL_LSTM,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )


class GRUClassifier(BaseTemporalClassifier):
    """Gated Recurrent Unit (GRU) observable behaviour classifier."""

    def __init__(
        self,
        input_size: int = CNN_FEATURE_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int = NUM_TARGET_CLASSES,
    ):
        super().__init__(
            model_type=MODEL_GRU,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )


def create_temporal_model(
    model_type: str,
    input_size: int = CNN_FEATURE_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    num_classes: int = NUM_TARGET_CLASSES,
) -> BaseTemporalClassifier:
    """Factory function for instantiating temporal models."""
    norm_type = model_type.lower().strip()
    if norm_type == MODEL_RNN:
        return RNNClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
    elif norm_type == MODEL_LSTM:
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
    elif norm_type == MODEL_GRU:
        return GRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Choose from {SUPPORTED_TEMPORAL_MODELS}."
        )
