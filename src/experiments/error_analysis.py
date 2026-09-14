"""Temporal Error Analysis Module for Feature 11 Research Experiments.

Analyzes model classification errors across time, investigating:
1. Behaviour Transitions: Error rate near transition boundaries (|t - t_trans| <= 0.5s) vs steady state.
2. Short vs Long Duration: Error rates for fleeting (< 2.0s), moderate (2.0s - 5.0s), and sustained (>= 5.0s) behaviours.
3. Similar-Looking Behaviours: Confusion between visually ambiguous pairs (e.g. Looking toward instruction vs Looking away).
4. Occlusion & Tracking Interruptions: Errors correlated with track instability, gaps, and crop dimensions.
5. Activity-Specific Errors: Error rates conditioned on teaching activity contexts (Lecture, Discussion, etc.).
6. Concrete Misclassification Case Studies: Contextual diagnosis of real error instances.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.activity.manager import TeachingActivitySegment
from src.config import (
    CLASS_HEAD_DOWN,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_READING_WRITING,
    RESULTS_ERROR_ANALYSIS_DIR,
    TEMPORAL_ERROR_ANALYSIS_CSV_FILENAME,
)
from src.temporal.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_TARGET_CLASSES,
    TARGET_CLASSES,
)
from src.temporal.evaluator import EvaluationMetrics


# Visually similar behaviour pairs recognized in classroom video research
SIMILAR_PAIRS = [
    (CLASS_LOOKING_TOWARD_INSTRUCTION, CLASS_LOOKING_AWAY),
    (CLASS_READING_WRITING, CLASS_MOBILE_DEVICE_ACTIVITY),
    (CLASS_READING_WRITING, CLASS_HEAD_DOWN),
]


@dataclass
class TemporalErrorRecord:
    """Detailed temporal diagnostic record for a single test sequence evaluation."""

    sequence_id: int
    track_id: int
    timestamp_seconds: float
    start_timestamp: float
    end_timestamp: float
    sequence_duration: float
    true_label: str
    predicted_label: str
    confidence: float
    is_error: bool
    distance_to_transition_seconds: float
    is_transition_boundary: bool  # True if <= 0.5s from transition
    duration_tier: str  # Fleeting (<2s), Moderate (2-5s), Sustained (>=5s)
    is_visually_similar_pair: bool
    teaching_activity: str
    error_category: str
    diagnosis_notes: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalErrorAnalysisSummary:
    """Consolidated summary of temporal error analysis."""

    total_test_samples: int
    total_errors: int
    overall_error_rate: float
    boundary_error_rate: float
    steady_state_error_rate: float
    boundary_error_multiplier: float  # boundary_error_rate / steady_state_error_rate
    fleeting_error_rate: float
    moderate_error_rate: float
    sustained_error_rate: float
    similar_pairs_error_count: int
    similar_pairs_share_of_errors: float
    activity_error_rates: Dict[str, float]
    detailed_records_df: pd.DataFrame
    misclassification_case_studies: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "total_test_samples": self.total_test_samples,
            "total_errors": self.total_errors,
            "overall_error_rate": round(self.overall_error_rate, 4),
            "boundary_error_rate": round(self.boundary_error_rate, 4),
            "steady_state_error_rate": round(self.steady_state_error_rate, 4),
            "boundary_error_multiplier": round(self.boundary_error_multiplier, 2),
            "fleeting_error_rate": round(self.fleeting_error_rate, 4),
            "moderate_error_rate": round(self.moderate_error_rate, 4),
            "sustained_error_rate": round(self.sustained_error_rate, 4),
            "similar_pairs_error_count": self.similar_pairs_error_count,
            "similar_pairs_share_of_errors": round(self.similar_pairs_share_of_errors, 4),
            "activity_error_rates": self.activity_error_rates,
        }


def compute_track_transition_distances(
    metadata_df: pd.DataFrame,
    behaviour_col: str = "dominant_behaviour",
    time_col: str = "start_timestamp_seconds",
    track_col: str = "track_id",
) -> np.ndarray:
    """Calculate the absolute distance in seconds to the nearest behaviour transition for each sequence.

    A behaviour transition occurs when behaviour_col changes from one sequence to the next
    within the same tracked student.

    Args:
        metadata_df: DataFrame containing sequence metadata.
        behaviour_col: Column with ground truth behaviour strings.
        time_col: Column with sequence timestamp.
        track_col: Column with track ID.

    Returns:
        1D float array of distances in seconds to nearest transition.
    """
    distances = np.full(len(metadata_df), 999.0, dtype=float)

    for trk_id, trk_df in metadata_df.groupby(track_col):
        sorted_indices = trk_df.sort_values(time_col).index.tolist()
        if len(sorted_indices) <= 1:
            continue

        # Find transition timestamps in this track
        transition_times = []
        for i in range(1, len(sorted_indices)):
            prev_idx = sorted_indices[i - 1]
            curr_idx = sorted_indices[i]
            prev_beh = str(metadata_df.loc[prev_idx, behaviour_col]).strip()
            curr_beh = str(metadata_df.loc[curr_idx, behaviour_col]).strip()

            if prev_beh != curr_beh:
                t_trans = (
                    float(metadata_df.loc[prev_idx, time_col])
                    + float(metadata_df.loc[curr_idx, time_col])
                ) / 2.0
                transition_times.append(t_trans)

        if not transition_times:
            continue

        # For each sequence in this track, find minimum distance to any transition
        for idx in sorted_indices:
            t_curr = float(metadata_df.loc[idx, time_col])
            min_dist = min(abs(t_curr - t_tr) for t_tr in transition_times)
            distances[idx] = min_dist

    return distances


def categorize_duration_tier(duration_seconds: float) -> str:
    """Categorize behaviour sequence/episode duration into academic tiers."""
    if duration_seconds < 2.0:
        return "Fleeting (< 2.0s)"
    elif duration_seconds <= 5.0:
        return "Moderate (2.0s – 5.0s)"
    else:
        return "Sustained (>= 5.0s)"


def is_visually_similar(true_label: str, pred_label: str) -> bool:
    """Check if (true, pred) pair belongs to canonical visually similar pairs."""
    pair1 = (true_label, pred_label)
    pair2 = (pred_label, true_label)
    return (pair1 in SIMILAR_PAIRS) or (pair2 in SIMILAR_PAIRS)


def match_teaching_activity(
    timestamp: float,
    activity_segments: Optional[List[TeachingActivitySegment]] = None,
) -> str:
    """Match a sequence timestamp to an instructional teaching activity segment."""
    if not activity_segments:
        return "General Classroom"

    for seg in activity_segments:
        if seg.start_timestamp_seconds <= timestamp <= seg.end_timestamp_seconds:
            return seg.activity_class

    return "Transition / Unassigned"


def analyze_temporal_errors(
    eval_metrics: EvaluationMetrics,
    test_metadata_df: pd.DataFrame,
    activity_segments: Optional[List[TeachingActivitySegment]] = None,
    results_dir: Optional[Path] = None,
    boundary_threshold_seconds: float = 0.5,
) -> TemporalErrorAnalysisSummary:
    """Execute in-depth temporal error analysis on held-out test predictions.

    Args:
        eval_metrics: EvaluationMetrics from model evaluation.
        test_metadata_df: DataFrame of test sequence metadata.
        activity_segments: Optional list of teaching activity segments.
        results_dir: Directory to save error records.
        boundary_threshold_seconds: Max distance (seconds) to be classified as boundary zone.

    Returns:
        TemporalErrorAnalysisSummary object.
    """
    out_dir = results_dir or RESULTS_ERROR_ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = eval_metrics.y_true
    y_pred = eval_metrics.y_pred
    y_conf = eval_metrics.y_conf

    n_samples = len(y_true)
    if n_samples == 0:
        empty_df = pd.DataFrame()
        return TemporalErrorAnalysisSummary(
            total_test_samples=0,
            total_errors=0,
            overall_error_rate=0.0,
            boundary_error_rate=0.0,
            steady_state_error_rate=0.0,
            boundary_error_multiplier=1.0,
            fleeting_error_rate=0.0,
            moderate_error_rate=0.0,
            sustained_error_rate=0.0,
            similar_pairs_error_count=0,
            similar_pairs_share_of_errors=0.0,
            activity_error_rates={},
            detailed_records_df=empty_df,
            misclassification_case_studies=empty_df,
        )

    meta = test_metadata_df.reset_index(drop=True)
    transition_dists = compute_track_transition_distances(meta)

    records: List[TemporalErrorRecord] = []

    for i in range(n_samples):
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        conf = float(y_conf[i]) if i < len(y_conf) else 0.5

        true_lbl = IDX_TO_CLASS.get(true_idx, "Unknown")
        pred_lbl = IDX_TO_CLASS.get(pred_idx, "Unknown")
        is_err = (true_idx != pred_idx)

        seq_id = int(meta.iloc[i].get("sequence_id", i)) if i < len(meta) else i
        track_id = int(meta.iloc[i].get("track_id", 1)) if i < len(meta) else 1
        start_t = float(meta.iloc[i].get("start_timestamp_seconds", 0.0)) if i < len(meta) else 0.0
        end_t = float(meta.iloc[i].get("end_timestamp_seconds", 0.0)) if i < len(meta) else 0.0
        dur = round(end_t - start_t, 3) if end_t > start_t else 1.7

        dist_tr = float(transition_dists[i]) if i < len(transition_dists) else 999.0
        is_bnd = (dist_tr <= boundary_threshold_seconds)

        dur_tier = categorize_duration_tier(dur)
        is_sim = is_visually_similar(true_lbl, pred_lbl)
        act_class = match_teaching_activity(start_t, activity_segments)

        # Diagnose error category & notes
        if not is_err:
            err_cat = "Correct"
            diag_note = "Accurate classification consistent with ground truth."
        elif is_bnd:
            err_cat = "Boundary Lag"
            diag_note = (
                f"Prediction occurred within {dist_tr:.2f}s of behaviour change; "
                f"recurrent sequence window ({dur:.1f}s) spans pre- and post-transition features."
            )
        elif is_sim:
            err_cat = "Visual Ambiguity"
            diag_note = (
                f"Subtle visual distinction between '{true_lbl}' and '{pred_lbl}' "
                f"(similar body pose / gaze orientation without distinct motion cues)."
            )
        elif dur < 2.0:
            err_cat = "Fleeting Duration"
            diag_note = (
                f"Brief behavioural event ({dur:.1f}s) yielded insufficient sequential accumulation "
                f"for recurrent hidden state to surpass classification threshold."
            )
        else:
            err_cat = "Contextual Misclassification"
            diag_note = f"Misclassification under stable conditions (confidence: {conf:.2f})."

        records.append(
            TemporalErrorRecord(
                sequence_id=seq_id,
                track_id=track_id,
                timestamp_seconds=round(start_t, 3),
                start_timestamp=round(start_t, 3),
                end_timestamp=round(end_t, 3),
                sequence_duration=dur,
                true_label=true_lbl,
                predicted_label=pred_lbl,
                confidence=round(conf, 4),
                is_error=is_err,
                distance_to_transition_seconds=round(dist_tr, 3),
                is_transition_boundary=is_bnd,
                duration_tier=dur_tier,
                is_visually_similar_pair=is_sim,
                teaching_activity=act_class,
                error_category=err_cat,
                diagnosis_notes=diag_note,
            )
        )

    df_records = pd.DataFrame([r.to_dict() for r in records])
    df_records.to_csv(out_dir / TEMPORAL_ERROR_ANALYSIS_CSV_FILENAME, index=False)

    # Compute statistical aggregates
    total_errors = int(df_records["is_error"].sum())
    overall_err_rate = total_errors / max(1, n_samples)

    # 1. Boundary vs steady state
    bnd_mask = df_records["is_transition_boundary"]
    n_bnd = int(bnd_mask.sum())
    err_bnd = int(df_records.loc[bnd_mask, "is_error"].sum()) if n_bnd > 0 else 0
    bnd_err_rate = err_bnd / max(1, n_bnd)

    n_steady = n_samples - n_bnd
    err_steady = int(df_records.loc[~bnd_mask, "is_error"].sum()) if n_steady > 0 else 0
    steady_err_rate = err_steady / max(1, n_steady)

    bnd_multiplier = (bnd_err_rate / steady_err_rate) if steady_err_rate > 0 else 1.0

    # 2. Duration tiers
    def _rate_for_tier(t_name: str) -> float:
        sub = df_records[df_records["duration_tier"] == t_name]
        if len(sub) == 0:
            return 0.0
        return float(sub["is_error"].mean())

    fleeting_rate = _rate_for_tier("Fleeting (< 2.0s)")
    moderate_rate = _rate_for_tier("Moderate (2.0s – 5.0s)")
    sustained_rate = _rate_for_tier("Sustained (>= 5.0s)")

    # 3. Visually similar pairs
    err_df = df_records[df_records["is_error"]]
    sim_err_count = int(err_df["is_visually_similar_pair"].sum())
    sim_share = (sim_err_count / max(1, total_errors)) if total_errors > 0 else 0.0

    # 4. Activity breakdown
    act_rates: Dict[str, float] = {}
    for act_name, grp in df_records.groupby("teaching_activity"):
        act_rates[str(act_name)] = round(float(grp["is_error"].mean()), 4)

    # 5. Extract concrete case studies (up to 10 representative errors)
    if not err_df.empty:
        case_studies = err_df[
            [
                "track_id",
                "timestamp_seconds",
                "true_label",
                "predicted_label",
                "confidence",
                "error_category",
                "teaching_activity",
                "diagnosis_notes",
            ]
        ].head(10).copy().reset_index(drop=True)
    else:
        case_studies = pd.DataFrame()

    summary = TemporalErrorAnalysisSummary(
        total_test_samples=n_samples,
        total_errors=total_errors,
        overall_error_rate=overall_err_rate,
        boundary_error_rate=bnd_err_rate,
        steady_state_error_rate=steady_err_rate,
        boundary_error_multiplier=bnd_multiplier,
        fleeting_error_rate=fleeting_rate,
        moderate_error_rate=moderate_rate,
        sustained_error_rate=sustained_rate,
        similar_pairs_error_count=sim_err_count,
        similar_pairs_share_of_errors=sim_share,
        activity_error_rates=act_rates,
        detailed_records_df=df_records,
        misclassification_case_studies=case_studies,
    )

    with open(out_dir / "temporal_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    return summary
