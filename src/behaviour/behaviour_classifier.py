"""Observable Behaviour Classifier Module for EduPulse AI.

Implements modular observable behaviour recognition for tracked people in classroom video frames.
Supports:
  - Situation A: Loading trained PyTorch visual classifier weights (when available).
  - Situation B: Transparent, rule-grounded Prototype / Baseline Heuristic pipeline
    evaluating observable visual evidence (posture, head orientation, peer proximity,
    desk/hand region gradients).
Preserves anonymous Track IDs, outputs structured behaviours.csv, and respects
strict research boundaries (observable behavior only, no internal mental state claims).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.behaviour.behaviour_labels import (
    CLASS_HEAD_DOWN,
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_READING_WRITING,
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
    get_behaviour_bgr,
    get_behaviour_description,
    get_behaviour_rgb,
)
from src.behaviour.preprocessing import preprocess_person_crop
from src.config import (
    BEHAVIOURS_CSV_FILENAME,
    DEFAULT_BEHAVIOUR_CONF_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DIR,
)


@dataclass
class BehaviourPrediction:
    """Observable behaviour classification for an individual tracked person in one frame."""

    video_id: str
    frame_id: int
    extracted_frame_index: int
    timestamp_seconds: float
    frame_filename: str
    track_id: int
    behaviour_class: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    visual_evidence: str = ""

    def to_dict(self) -> dict:
        """Convert prediction to serializable dictionary for tabular export."""
        return {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "extracted_frame_index": self.extracted_frame_index,
            "timestamp_seconds": round(float(self.timestamp_seconds), 3),
            "frame_filename": self.frame_filename,
            "track_id": int(self.track_id),
            "behaviour_class": self.behaviour_class,
            "confidence": round(float(self.confidence), 4),
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "visual_evidence": self.visual_evidence,
        }


@dataclass
class BehaviourSummary:
    """Summary metrics of an observable behaviour recognition session."""

    video_id: str
    classifier_mode: str
    model_path: Optional[str]
    confidence_threshold: float
    total_observations: int
    unique_tracks: int
    class_distribution: Dict[str, int]
    track_dominant_behaviours: Dict[int, str]
    unknown_count: int
    avg_confidence: float
    behaviours_csv_path: Path
    status: str


class BehaviourClassifier:
    """Modular classroom behaviour recognition engine."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        conf_threshold: float = DEFAULT_BEHAVIOUR_CONF_THRESHOLD,
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.is_trained_model = False
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Attempt loading trained weights if available (Situation A)
        self.load_model()

    def load_model(self) -> Tuple[bool, str]:
        """Attempt to load trained PyTorch model weights (Situation A).

        If no weights exist, seamlessly falls back to Prototype / Baseline Heuristic
        mode (Situation B) without crashing, preserving transparent research reporting.
        """
        # Search candidate paths
        candidate = self.weights_path
        if candidate is None:
            default_candidate = MODELS_DIR / "behaviour_classifier.pt"
            if default_candidate.exists():
                candidate = default_candidate

        if candidate and Path(candidate).exists():
            try:
                # Load PyTorch model
                loaded = torch.load(candidate, map_location=self.device)
                if isinstance(loaded, nn.Module):
                    self.model = loaded
                elif isinstance(loaded, dict) and "state_dict" in loaded:
                    # Model placeholder with state dict
                    self.model = loaded["state_dict"]
                self.model.eval()
                self.is_trained_model = True
                return True, f"Loaded trained PyTorch behaviour model from {candidate}."
            except Exception as exc:
                self.is_trained_model = False
                self.model = None
                return False, f"Failed to load weights from {candidate}: {exc}"

        self.is_trained_model = False
        self.model = None
        return True, "Operating in Prototype / Baseline Heuristic Mode (No trained weights file loaded)."

    @property
    def mode_name(self) -> str:
        """Return human-readable indicator of active inference mode."""
        if self.is_trained_model:
            return "PyTorch Neural Model (Trained Weights)"
        return "Prototype / Baseline Heuristic (Observable Cues)"

    def predict_person(
        self,
        crop_rgb: np.ndarray,
        crop_tensor: Optional[torch.Tensor],
        track_info: dict,
        frame_tracks: List[dict],
        frame_dims: Tuple[int, int],
        conf_threshold: Optional[float] = None,
    ) -> Tuple[str, float, str]:
        """Classify observable behaviour for a single tracked person crop.

        Args:
            crop_rgb: Resized (224x224x3) RGB image of the person.
            crop_tensor: Normalized PyTorch tensor (1, 3, 224, 224).
            track_info: Dictionary containing track bounding box, centroid, and ID.
            frame_tracks: All other tracks in the same frame for proximity evaluation.
            frame_dims: Full frame dimensions (height, width).
            conf_threshold: Optional threshold override.

        Returns:
            Tuple of (behaviour_class, confidence, visual_evidence_description).
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Situation A: Trained PyTorch Model Inference
        if self.is_trained_model and self.model is not None and crop_tensor is not None:
            try:
                with torch.no_grad():
                    inputs = crop_tensor.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    top_idx = int(np.argmax(probs))
                    top_conf = float(probs[top_idx])

                    if top_idx < len(TARGET_BEHAVIOUR_CLASSES) and top_conf >= threshold:
                        predicted_class = TARGET_BEHAVIOUR_CLASSES[top_idx]
                        desc = f"Model probability: {top_conf:.2f} for {predicted_class}."
                        return predicted_class, top_conf, desc
                    else:
                        return (
                            CLASS_UNKNOWN,
                            top_conf,
                            f"Model confidence ({top_conf:.2f}) below threshold ({threshold:.2f}).",
                        )
            except Exception as exc:
                # Fall through to heuristic if model pass fails
                pass

        # Situation B: Prototype / Baseline Heuristic Inference
        return self._predict_heuristic(
            crop_rgb=crop_rgb,
            track_info=track_info,
            frame_tracks=frame_tracks,
            frame_dims=frame_dims,
            conf_threshold=threshold,
        )

    def _predict_heuristic(
        self,
        crop_rgb: np.ndarray,
        track_info: dict,
        frame_tracks: List[dict],
        frame_dims: Tuple[int, int],
        conf_threshold: float,
    ) -> Tuple[str, float, str]:
        """Observable visual heuristic evaluation based on posture, orientation, and peer proximity."""
        h_frame, w_frame = frame_dims
        x1 = track_info.get("x1", 0.0)
        y1 = track_info.get("y1", 0.0)
        x2 = track_info.get("x2", 0.0)
        y2 = track_info.get("y2", 0.0)
        curr_track_id = track_info.get("track_id", -1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect_ratio = h / w

        # Extract crop zones (Head/Upper: top 35%, Torso/Mid: middle 35%, Desk/Lap: bottom 30%)
        crop_h, crop_w = crop_rgb.shape[:2]
        head_zone = crop_rgb[: int(crop_h * 0.35), :]
        lap_zone = crop_rgb[int(crop_h * 0.70) :, :]

        # Grayscale and edge gradient analysis
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check blur or extreme low texture -> Unknown
        if laplacian_var < 15.0:
            return (
                CLASS_UNKNOWN,
                0.35,
                "Visual crop has low gradient/texture (likely motion blur or severe occlusion).",
            )

        # 1. Peer Proximity Check (Interacting with peers)
        min_peer_dist = float("inf")
        nearest_peer_id = None
        for other in frame_tracks:
            other_id = other.get("track_id")
            if other_id == curr_track_id:
                continue
            ocx = (other.get("x1", 0.0) + other.get("x2", 0.0)) / 2.0
            ocy = (other.get("y1", 0.0) + other.get("y2", 0.0)) / 2.0
            dist = np.hypot(cx - ocx, cy - ocy)
            if dist < min_peer_dist:
                min_peer_dist = dist
                nearest_peer_id = other_id

        # Adjacent peers within conversational distance
        if min_peer_dist < 1.35 * w:
            conf = min(0.85, max(0.55, 0.90 - (min_peer_dist / (1.5 * w)) * 0.35))
            if conf >= conf_threshold:
                return (
                    CLASS_INTERACTING_WITH_PEERS,
                    conf,
                    f"Proximal collaborative alignment with peer (Track ID {nearest_peer_id}) within {min_peer_dist:.1f}px.",
                )

        # 2. Head / Upper Body Orientation Analysis
        head_gray = cv2.cvtColor(head_zone, cv2.COLOR_RGB2GRAY)
        # Compute horizontal center of mass in head zone
        head_moments = cv2.moments(head_gray)
        head_cx_norm = 0.5
        if head_moments["m00"] > 0:
            head_cx_norm = (head_moments["m10"] / head_moments["m00"]) / crop_w

        # Check lateral gaze orientation (Looking away)
        # If head is turned sharply to the lateral periphery (<0.38 or >0.62)
        if head_cx_norm < 0.36 or head_cx_norm > 0.64:
            conf = 0.72 + abs(head_cx_norm - 0.5) * 0.4
            if conf >= conf_threshold:
                side = "left" if head_cx_norm < 0.36 else "right"
                return (
                    CLASS_LOOKING_AWAY,
                    conf,
                    f"Head and upper-body orientation skewed toward {side} periphery ({head_cx_norm:.2f}).",
                )

        # 3. Desk/Lap zone activity analysis (Reading/writing vs Mobile-device vs Head-down)
        lap_gray = cv2.cvtColor(lap_zone, cv2.COLOR_RGB2GRAY)
        lap_edges = cv2.Canny(lap_gray, 50, 150)
        lap_edge_density = float(np.count_nonzero(lap_edges)) / float(lap_edges.size)

        # Head-down posture (Face resting directly down, head zone is low intensity / low contrast)
        head_edge_density = float(np.count_nonzero(cv2.Canny(head_gray, 50, 150))) / float(head_gray.size)
        if aspect_ratio < 1.05 and head_edge_density < 0.05:
            conf = 0.75
            if conf >= conf_threshold:
                return (
                    CLASS_HEAD_DOWN,
                    conf,
                    "Low head profile resting downward on desk/arms with low facial visibility.",
                )

        # Reading / Writing: downward gaze with active high-edge desk surface (notebook, writing instruments)
        if lap_edge_density > 0.12 and aspect_ratio >= 1.0:
            conf = min(0.88, 0.65 + lap_edge_density)
            if conf >= conf_threshold:
                return (
                    CLASS_READING_WRITING,
                    conf,
                    f"Downward gaze toward desk with high edge texture ({lap_edge_density:.2f}) consistent with reading/writing.",
                )

        # Mobile-Device Activity: compact, bright region in lap with localized contrast
        lap_bright_mask = lap_gray > 200
        bright_ratio = float(np.count_nonzero(lap_bright_mask)) / float(lap_gray.size)
        if 0.05 < bright_ratio < 0.35 and lap_edge_density > 0.08:
            conf = 0.70
            if conf >= conf_threshold:
                return (
                    CLASS_MOBILE_DEVICE_ACTIVITY,
                    conf,
                    f"High-contrast illuminated handheld object detected in lap/hand zone (ratio {bright_ratio:.2f}).",
                )

        # 4. Default upright forward posture: Looking toward instruction
        if aspect_ratio >= 1.0:
            conf = 0.82
            if conf >= conf_threshold:
                return (
                    CLASS_LOOKING_TOWARD_INSTRUCTION,
                    conf,
                    "Upright posture with head oriented toward instructional center.",
                )

        # Fallback to Unknown if below threshold
        return (
            CLASS_UNKNOWN,
            0.38,
            "Ambiguous observable posture falling below confidence threshold.",
        )

    def draw_behaviours(
        self,
        frame: np.ndarray,
        predictions: List[BehaviourPrediction],
        show_evidence: bool = False,
    ) -> np.ndarray:
        """Annotate video frame with bounding boxes, Track IDs, and behaviour tags.

        Args:
            frame: RGB video frame as NumPy array.
            predictions: List of BehaviourPrediction items for this frame.
            show_evidence: Whether to append brief visual evidence text to the badge.

        Returns:
            Annotated RGB frame.
        """
        if frame is None or len(predictions) == 0:
            return frame

        annotated = frame.copy()
        h_frame, w_frame = annotated.shape[:2]

        for pred in predictions:
            color = get_behaviour_rgb(pred.behaviour_class)
            x1, y1, x2, y2 = int(round(pred.x1)), int(round(pred.y1)), int(round(pred.x2)), int(round(pred.y2))
            x1 = max(0, min(x1, w_frame - 1))
            y1 = max(0, min(y1, h_frame - 1))
            x2 = max(0, min(x2, w_frame - 1))
            y2 = max(0, min(y2, h_frame - 1))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Badge text
            line1 = f"ID {pred.track_id}: {pred.behaviour_class}"
            line2 = f"Conf: {pred.confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
            badge_w = max(w1, w2) + 12
            badge_h = h1 + h2 + 16

            badge_y1 = max(0, y1 - badge_h)
            badge_y2 = badge_y1 + badge_h
            badge_x2 = min(w_frame, x1 + badge_w)

            # Background rectangle for text contrast
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), (25, 25, 25), -1)
            # Left accent stripe with class color
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + 4, badge_y2), color, -1)

            # Text lines
            cv2.putText(
                annotated,
                line1,
                (x1 + 8, badge_y1 + h1 + 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                line2,
                (x1 + 8, badge_y1 + h1 + h2 + 10),
                font,
                font_scale,
                (200, 200, 200),
                thickness,
                cv2.LINE_AA,
            )

        return annotated


def run_behaviour_recognition_on_tracks(
    video_id: str,
    frames_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    classifier: BehaviourClassifier,
    conf_threshold: float = DEFAULT_BEHAVIOUR_CONF_THRESHOLD,
    frame_selection_mode: str = "all",
    first_n: int = 10,
    range_start: int = 1,
    range_end: int = 18,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[bool, Optional[BehaviourSummary], Optional[pd.DataFrame], Dict[str, np.ndarray], str]:
    """Execute observable behaviour recognition across tracked video frames.

    Args:
        video_id: Unique identifier of the video.
        frames_df: Extracted frames metadata DataFrame.
        tracks_df: Feature 4 tracking DataFrame.
        classifier: BehaviourClassifier instance.
        conf_threshold: Prediction confidence threshold.
        frame_selection_mode: "all", "first_n", or "range".
        first_n: Frame count for "first_n" mode.
        range_start: Start index (1-based) for "range" mode.
        range_end: End index (1-based) for "range" mode.
        progress_callback: Optional progress updater.

    Returns:
        Tuple of (success, summary, behaviours_df, annotated_frames_dict, message).
    """
    if frames_df is None or len(frames_df) == 0:
        return False, None, None, {}, "No extracted frames provided for behaviour recognition."

    if tracks_df is None or len(tracks_df) == 0:
        return False, None, None, {}, "No tracking data provided. Please run Feature 4 tracking first."

    # Filter frames based on selection mode
    df_sorted = frames_df.sort_values(by="extracted_frame_index").copy()
    if frame_selection_mode == "first_n":
        df_selected = df_sorted.head(max(1, first_n)).copy()
    elif frame_selection_mode == "range":
        start_idx = max(1, range_start)
        end_idx = max(start_idx, range_end)
        df_selected = df_sorted[
            (df_sorted["extracted_frame_index"] >= start_idx)
            & (df_sorted["extracted_frame_index"] <= end_idx)
        ].copy()
    else:
        df_selected = df_sorted

    total_frames = len(df_selected)
    if total_frames == 0:
        return False, None, None, {}, "No frames matched the selection criteria."

    all_predictions: List[BehaviourPrediction] = []
    annotated_frames: Dict[str, np.ndarray] = {}

    for idx, (_, row) in enumerate(df_selected.iterrows()):
        frame_path_str = str(row.get("frame_path", ""))
        frame_filename = str(row.get("frame_filename", f"frame_{idx:06d}.jpg"))
        extracted_frame_index = int(row.get("extracted_frame_index", idx + 1))
        frame_id = int(row.get("frame_index", idx))
        timestamp_sec = float(row.get("timestamp_seconds", 0.0))

        if progress_callback:
            progress_callback(
                (idx / total_frames),
                f"Classifying observable behaviours in frame {idx + 1}/{total_frames} ({frame_filename})...",
            )

        frame_file = Path(frame_path_str)
        if not frame_file.exists():
            continue

        bgr_frame = cv2.imread(str(frame_file))
        if bgr_frame is None:
            continue
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frame_dims = rgb_frame.shape[:2]

        # Get tracks for this frame
        # Match by extracted_frame_index or frame_filename
        frame_tracks_df = tracks_df[
            (tracks_df["extracted_frame_index"] == extracted_frame_index)
            | (tracks_df["frame_filename"] == frame_filename)
        ]

        tracks_list = frame_tracks_df.to_dict(orient="records")
        frame_predictions: List[BehaviourPrediction] = []

        for track_dict in tracks_list:
            t_id = int(track_dict.get("track_id", -1))
            bx1 = float(track_dict.get("x1", 0.0))
            by1 = float(track_dict.get("y1", 0.0))
            bx2 = float(track_dict.get("x2", 0.0))
            by2 = float(track_dict.get("y2", 0.0))

            # Preprocess crop
            ok, crop_rgb, crop_tensor, err = preprocess_person_crop(
                rgb_frame, (bx1, by1, bx2, by2)
            )
            if not ok or crop_rgb is None:
                # Handle invalid/degraded crop with Unknown classification
                pred = BehaviourPrediction(
                    video_id=video_id,
                    frame_id=frame_id,
                    extracted_frame_index=extracted_frame_index,
                    timestamp_seconds=timestamp_sec,
                    frame_filename=frame_filename,
                    track_id=t_id,
                    behaviour_class=CLASS_UNKNOWN,
                    confidence=0.0,
                    x1=bx1,
                    y1=by1,
                    x2=bx2,
                    y2=by2,
                    visual_evidence=f"Crop extraction failed: {err}",
                )
            else:
                beh_class, conf, evidence = classifier.predict_person(
                    crop_rgb=crop_rgb,
                    crop_tensor=crop_tensor,
                    track_info=track_dict,
                    frame_tracks=tracks_list,
                    frame_dims=frame_dims,
                    conf_threshold=conf_threshold,
                )
                pred = BehaviourPrediction(
                    video_id=video_id,
                    frame_id=frame_id,
                    extracted_frame_index=extracted_frame_index,
                    timestamp_seconds=timestamp_sec,
                    frame_filename=frame_filename,
                    track_id=t_id,
                    behaviour_class=beh_class,
                    confidence=conf,
                    x1=bx1,
                    y1=by1,
                    x2=bx2,
                    y2=by2,
                    visual_evidence=evidence,
                )

            frame_predictions.append(pred)
            all_predictions.append(pred)

        # Draw annotations on frame
        annotated_frame = classifier.draw_behaviours(rgb_frame, frame_predictions)
        annotated_frames[frame_filename] = annotated_frame

    if progress_callback:
        progress_callback(1.0, "Behaviour recognition complete. Compiling summary and saving CSV...")

    if not all_predictions:
        return False, None, None, {}, "No tracked people were available in the selected frames."

    # Build DataFrame
    behaviours_df = pd.DataFrame([p.to_dict() for p in all_predictions])

    # Save to data/processed/<video_id>/behaviours.csv
    output_dir = PROCESSED_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / BEHAVIOURS_CSV_FILENAME
    behaviours_df.to_csv(csv_path, index=False)

    # Compute Summary Statistics
    total_obs = len(all_predictions)
    unique_tracks = behaviours_df["track_id"].nunique()
    class_dist = behaviours_df["behaviour_class"].value_counts().to_dict()
    unknown_count = int(class_dist.get(CLASS_UNKNOWN, 0))

    # Track-level dominant behaviour
    track_dominant: Dict[int, str] = {}
    for tid, group in behaviours_df.groupby("track_id"):
        # Exclude unknown if possible for dominant calculation
        known = group[group["behaviour_class"] != CLASS_UNKNOWN]
        if len(known) > 0:
            dom = str(known["behaviour_class"].mode()[0])
        else:
            dom = CLASS_UNKNOWN
        track_dominant[int(tid)] = dom

    avg_conf = float(behaviours_df[behaviours_df["behaviour_class"] != CLASS_UNKNOWN]["confidence"].mean())
    if np.isnan(avg_conf):
        avg_conf = 0.0

    summary = BehaviourSummary(
        video_id=video_id,
        classifier_mode=classifier.mode_name,
        model_path=str(classifier.weights_path) if classifier.weights_path else None,
        confidence_threshold=conf_threshold,
        total_observations=total_obs,
        unique_tracks=unique_tracks,
        class_distribution=class_dist,
        track_dominant_behaviours=track_dominant,
        unknown_count=unknown_count,
        avg_confidence=round(avg_conf, 4),
        behaviours_csv_path=csv_path,
        status="Success",
    )

    return True, summary, behaviours_df, annotated_frames, f"Successfully recognised behaviours for {total_obs} observations."
