"""Visual Analytics Module for Feature 11 Research Experiments.

Creates high-resolution publication-quality scientific figures:
1. plot_model_comparison: Grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models.
2. plot_confusion_matrices: Multi-panel 6x6 confusion matrix heatmaps.
3. plot_training_curves: Training and validation loss/accuracy convergence curves.
4. plot_per_class_f1: Grouped bar chart of F1-scores across all 6 observable behaviour classes.
5. plot_temporal_error_dynamics: Transition boundary, duration tier, and teaching activity error breakdown.
6. plot_ablation_summary: Waterfall / delta chart of empirical ablation impact.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.activity.activity_labels import get_activity_hex
from src.behaviour.behaviour_labels import get_behaviour_hex
from src.config import (
    ALL_COMPARISON_MODELS,
    BEHAVIOUR_COLORS,
    MODEL_FRAME_CNN,
    MODEL_GRU,
    MODEL_LSTM,
    MODEL_RNN,
    TARGET_OBSERVABLE_BEHAVIOURS,
)
from src.experiments.error_analysis import TemporalErrorAnalysisSummary
from src.temporal.dataset import NUM_TARGET_CLASSES, TARGET_CLASSES
from src.temporal.evaluator import EvaluationMetrics


# Canonical shortened labels for readable plots
SHORT_CLASS_NAMES = [
    "Looking Inst.",
    "Read/Write",
    "Peer Interact.",
    "Looking Away",
    "Mobile Device",
    "Head Down",
]

MODEL_PALETTE = {
    "Frame-level CNN": "#7F8C8D",  # Slate Gray
    "CNN + RNN": "#E67E22",        # Orange
    "CNN + LSTM": "#2980B9",       # Blue
    "CNN + GRU": "#27AE60",        # Green
}


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Create a publication-quality grouped bar chart comparing the 4 models across metrics."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    models = comparison_df["Model"].tolist()
    n_models = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / max(1, n_models)

    for i, model_name in enumerate(models):
        row = comparison_df[comparison_df["Model"] == model_name]
        vals = [float(row[m].values[0]) for m in metrics]
        color = MODEL_PALETTE.get(model_name, plt.cm.tab10(i))

        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.8)

        # Label values above bars
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.015,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_ylabel("Metric Score [0.0 – 1.0]", fontsize=11, fontweight="bold")
    ax.set_title("Observable Behaviour Recognition: 4-Way Model Performance Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    return fig


def plot_confusion_matrices(
    metrics_map: Dict[str, EvaluationMetrics],
    normalize: bool = True,
) -> plt.Figure:
    """Plot multi-panel 6x6 confusion matrix heatmaps for the evaluated models."""
    display_keys = [k for k in [MODEL_FRAME_CNN, MODEL_RNN, MODEL_LSTM, MODEL_GRU] if k in metrics_map]
    if not display_keys:
        display_keys = list(metrics_map.keys())

    n_panels = len(display_keys)
    cols = 2 if n_panels > 1 else 1
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.8 * rows), dpi=150)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    display_titles = {
        MODEL_FRAME_CNN: "Frame-level CNN (Static Baseline)",
        MODEL_RNN: "CNN + RNN",
        MODEL_LSTM: "CNN + LSTM",
        MODEL_GRU: "CNN + GRU",
    }

    labels = SHORT_CLASS_NAMES

    for idx, key in enumerate(display_keys):
        ax = axes[idx]
        metrics = metrics_map[key]
        cm = np.array(metrics.confusion_matrix)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            norm_cm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
            im = ax.imshow(norm_cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

        title = display_titles.get(key, key.upper())
        ax.set_title(f"{title} (F1: {metrics.macro_f1:.3f})", fontsize=11, fontweight="bold")

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate matrix cells
        thresh = 0.5 if normalize else (cm.max() / 2.0 if cm.max() > 0 else 1.0)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = norm_cm[r, c] if normalize else cm[r, c]
                raw_count = cm[r, c]
                text = f"{val:.1%}\n({raw_count})" if normalize else str(raw_count)
                text_color = "white" if (val > thresh and raw_count > 0) else "black"
                ax.text(c, r, text, ha="center", va="center", color=text_color, fontsize=7)

        ax.set_ylabel("True Observable Class", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted Observable Class", fontsize=9, fontweight="bold")

    # Hide unused subplots
    for idx in range(len(display_keys), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_training_curves(history_dfs: Dict[str, pd.DataFrame]) -> plt.Figure:
    """Plot multi-model training and validation convergence curves."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    model_colors = {
        MODEL_FRAME_CNN: "#7F8C8D",
        MODEL_RNN: "#E67E22",
        MODEL_LSTM: "#2980B9",
        MODEL_GRU: "#27AE60",
    }

    display_names = {
        MODEL_FRAME_CNN: "Frame-level CNN",
        MODEL_RNN: "CNN + RNN",
        MODEL_LSTM: "CNN + LSTM",
        MODEL_GRU: "CNN + GRU",
    }

    for m_key, df in history_dfs.items():
        if df.empty or "epoch" not in df.columns:
            continue

        c = model_colors.get(m_key, plt.cm.tab10(0))
        label = display_names.get(m_key, m_key.upper())

        epochs = df["epoch"]
        ax_loss.plot(epochs, df["train_loss"], label=f"{label} (Train)", color=c, linestyle=":", alpha=0.7)
        ax_loss.plot(epochs, df["val_loss"], label=f"{label} (Val)", color=c, linestyle="-", linewidth=2.0)

        ax_acc.plot(epochs, df["train_acc"], label=f"{label} (Train)", color=c, linestyle=":", alpha=0.7)
        ax_acc.plot(epochs, df["val_acc"], label=f"{label} (Val)", color=c, linestyle="-", linewidth=2.0)

    ax_loss.set_title("Cross-Entropy Loss Convergence", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.3)
    ax_loss.legend(fontsize=8, loc="upper right")

    ax_acc.set_title("Validation Accuracy Convergence", fontsize=11, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Accuracy", fontsize=10)
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(True, linestyle="--", alpha=0.3)
    ax_acc.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    return fig


def plot_per_class_f1(metrics_map: Dict[str, EvaluationMetrics]) -> plt.Figure:
    """Plot grouped bar chart of F1-scores across all 6 canonical behaviour classes."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)

    classes = TARGET_CLASSES
    short_labels = SHORT_CLASS_NAMES
    n_classes = len(classes)

    models_to_plot = [k for k in [MODEL_FRAME_CNN, MODEL_RNN, MODEL_LSTM, MODEL_GRU] if k in metrics_map]
    n_models = len(models_to_plot)

    x = np.arange(n_classes)
    width = 0.8 / max(1, n_models)

    display_names = {
        MODEL_FRAME_CNN: "Frame-level CNN",
        MODEL_RNN: "CNN + RNN",
        MODEL_LSTM: "CNN + LSTM",
        MODEL_GRU: "CNN + GRU",
    }

    for i, m_key in enumerate(models_to_plot):
        m = metrics_map[m_key]
        f1_vals = [float(m.per_class_metrics.get(cls, {}).get("f1", 0.0)) for cls in classes]
        color = MODEL_PALETTE.get(display_names.get(m_key, m_key), plt.cm.tab10(i))

        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, f1_vals, width, label=display_names.get(m_key, m_key), color=color, alpha=0.9, edgecolor="black", linewidth=0.7)

        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.015,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    ax.set_ylabel("F1-score [0.0 – 1.0]", fontsize=10, fontweight="bold")
    ax.set_title("Per-Class Observable Behaviour F1-Score Breakdown Across Architectures", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)

    plt.tight_layout()
    return fig


def plot_temporal_error_dynamics(error_summary: TemporalErrorAnalysisSummary) -> plt.Figure:
    """Plot multi-panel figure analyzing temporal error patterns across time."""
    fig, (ax_bnd, ax_dur, ax_act) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # 1. Boundary vs Steady-State
    bnd_labels = ["Boundary (≤ 0.5s)", "Steady-State (> 0.5s)"]
    bnd_rates = [
        error_summary.boundary_error_rate * 100.0,
        error_summary.steady_state_error_rate * 100.0,
    ]
    colors_bnd = ["#E74C3C", "#2ECC71"]
    b_bars = ax_bnd.bar(bnd_labels, bnd_rates, color=colors_bnd, width=0.5, edgecolor="black", linewidth=0.8)
    for b in b_bars:
        h = b.get_height()
        ax_bnd.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", fontweight="bold")
    ax_bnd.set_ylabel("Error Rate (%)", fontsize=10, fontweight="bold")
    ax_bnd.set_title(f"Transition Boundary Error\n({error_summary.boundary_error_multiplier:.1f}× Multiplier)", fontsize=11, fontweight="bold")
    ax_bnd.set_ylim(0.0, max(100.0, max(bnd_rates) + 15.0) if bnd_rates else 100.0)
    ax_bnd.grid(axis="y", linestyle="--", alpha=0.3)

    # 2. Duration Tiers
    dur_labels = ["Fleeting\n(< 2s)", "Moderate\n(2s – 5s)", "Sustained\n(≥ 5s)"]
    dur_rates = [
        error_summary.fleeting_error_rate * 100.0,
        error_summary.moderate_error_rate * 100.0,
        error_summary.sustained_error_rate * 100.0,
    ]
    d_bars = ax_dur.bar(dur_labels, dur_rates, color="#3498DB", width=0.55, edgecolor="black", linewidth=0.8)
    for b in d_bars:
        h = b.get_height()
        ax_dur.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", fontweight="bold")
    ax_dur.set_ylabel("Error Rate (%)", fontsize=10, fontweight="bold")
    ax_dur.set_title("Error Rate by Behaviour Duration", fontsize=11, fontweight="bold")
    ax_dur.set_ylim(0.0, max(100.0, max(dur_rates) + 15.0) if dur_rates else 100.0)
    ax_dur.grid(axis="y", linestyle="--", alpha=0.3)

    # 3. Teaching Activity Breakdown
    act_rates = error_summary.activity_error_rates
    if act_rates:
        acts = list(act_rates.keys())
        rates = [r * 100.0 for r in act_rates.values()]
        colors_act = [get_activity_hex(a) if a != "General Classroom" else "#95A5A6" for a in acts]
        a_bars = ax_act.bar(acts, rates, color=colors_act, width=0.55, edgecolor="black", linewidth=0.8)
        for b in a_bars:
            h = b.get_height()
            ax_act.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", fontweight="bold")
        ax_act.set_xticks(range(len(acts)))
        ax_act.set_xticklabels(acts, rotation=20, ha="right", fontsize=8)
    else:
        ax_act.text(0.5, 0.5, "No Activity Segments Defined", ha="center", va="center")

    ax_act.set_ylabel("Error Rate (%)", fontsize=10, fontweight="bold")
    ax_act.set_title("Activity-Conditioned Error Rates", fontsize=11, fontweight="bold")
    ax_act.set_ylim(0.0, 100.0)
    ax_act.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ablation_summary(ablation_df: pd.DataFrame) -> plt.Figure:
    """Create a delta chart visualizing the empirical impact of controlled ablations."""
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)

    if ablation_df.empty or "experimental_condition" not in ablation_df.columns:
        ax.text(0.5, 0.5, "No Ablation Trials Recorded", ha="center", va="center")
        return fig

    labels = ablation_df["experimental_condition"].tolist()
    deltas = ablation_df["macro_f1_delta"].astype(float).tolist()
    pcts = ablation_df["pct_change"].astype(float).tolist()

    y_pos = np.arange(len(labels))
    colors = ["#2ECC71" if d >= 0 else "#E74C3C" for d in deltas]

    bars = ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.8, height=0.55)
    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.0)

    for bar, pct in zip(bars, pcts):
        w = bar.get_width()
        x_text = w + (0.005 if w >= 0 else -0.005)
        ha = "left" if w >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2.0, f"{w:+.4f} ({pct:+.1f}%)", va="center", ha=ha, fontsize=8, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Macro F1-Score Change relative to Baseline (Δ F1)", fontsize=10, fontweight="bold")
    ax.set_title("Ablation Study: Empirical Impact on Observable Behaviour Classification", fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig
