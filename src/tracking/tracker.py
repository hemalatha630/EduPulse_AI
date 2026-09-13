"""Student / Person Tracking Module for EduPulse AI.

Implements multi-object tracking (ByteTrack / BoT-SORT) for classroom video frames.
Connects per-frame person detections into persistent, anonymous Track IDs across time,
generates spatial centroid trajectories, and saves structured tracking datasets.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_TRACKER,
    DEFAULT_TRACKING_CONF_THRESHOLD,
    DEFAULT_TRAJECTORY_MAX_POINTS,
    DEFAULT_YOLO_MODEL,
    MODELS_DIR,
    PERSON_CLASS_ID,
    PERSON_CLASS_NAME,
    PROCESSED_DIR,
    SUPPORTED_TRACKERS,
    TRACKS_CSV_FILENAME,
)
from src.preprocessing.frame_preprocessor import FramePreprocessor


@dataclass
class TrackResult:
    """Individual tracked person within a single video frame."""

    frame_id: int
    extracted_frame_index: int
    timestamp_seconds: float
    frame_filename: str
    track_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = PERSON_CLASS_ID
    class_name: str = PERSON_CLASS_NAME

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def to_dict(self) -> dict:
        """Convert track result to serializable dictionary for tabular export."""
        return {
            "frame_id": self.frame_id,
            "extracted_frame_index": self.extracted_frame_index,
            "timestamp_seconds": round(float(self.timestamp_seconds), 3),
            "frame_filename": self.frame_filename,
            "track_id": int(self.track_id),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "center_x": round(float(self.center_x), 2),
            "center_y": round(float(self.center_y), 2),
        }


@dataclass
class TrackingSummary:
    """Summary metrics of a multi-object tracking session."""

    video_id: str
    tracker_name: str
    confidence_threshold: float
    frames_processed: int
    unique_tracks: int
    avg_active_tracks_per_frame: float
    longest_track_duration_seconds: float
    longest_track_frames: int
    min_track_length: int
    max_track_length: int
    tracks_csv_path: Path
    status: str

    def to_dict(self) -> dict:
        """Convert tracking summary to dictionary."""
        data = asdict(self)
        data["tracks_csv_path"] = str(self.tracks_csv_path)
        return data


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a deterministic, distinct, vibrant RGB color for a given track ID."""
    # Use golden ratio hue distribution for maximum contrast between adjacent IDs
    hue = int((track_id * 137.508) % 180)  # OpenCV hue range [0, 179]
    hsv_pixel = np.uint8([[[hue, 220, 240]]])
    rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)
    r, g, b = rgb_pixel[0, 0]
    return int(r), int(g), int(b)


class PersonTracker:
    """Multi-object tracker for classroom video frames with persistent Track IDs."""

    def __init__(
        self,
        model_name: str = DEFAULT_YOLO_MODEL,
        tracker_type: str = DEFAULT_TRACKER,
        model_dir: Optional[Path] = None,
        max_trajectory_points: int = DEFAULT_TRAJECTORY_MAX_POINTS,
    ):
        """Initialize PersonTracker with model and tracker configurations."""
        self.model_name = model_name
        self.tracker_type = tracker_type if tracker_type in SUPPORTED_TRACKERS else DEFAULT_TRACKER
        self.tracker_yaml = f"{self.tracker_type}.yaml"
        self.model_dir = model_dir or MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / self.model_name
        self.max_trajectory_points = max_trajectory_points
        self._model = None

        # Cumulative track centroid trajectory history: {track_id: [(cx, cy), ...]}
        self.trajectory_history: Dict[int, List[Tuple[float, float]]] = {}

    def load_model(self) -> Tuple[bool, str]:
        """Load pretrained YOLO weights for tracking."""
        try:
            from ultralytics import YOLO

            target_path = self.model_path if self.model_path.exists() else self.model_name
            self._model = YOLO(str(target_path))

            # Ensure downloaded model is saved in models/ directory
            downloaded_weights = Path(f"{self.model_name}")
            if downloaded_weights.exists() and not self.model_path.exists():
                import shutil
                shutil.move(str(downloaded_weights), str(self.model_path))

            return True, f"Successfully loaded {self.model_name} with {self.tracker_type}"
        except Exception as exc:
            return False, f"Failed to load tracking model: {str(exc)}"

    def reset(self) -> None:
        """Reset internal tracker state and trajectory histories between runs."""
        self.trajectory_history.clear()
        if self._model is not None:
            # Recreate predictor to ensure clean state and reset track ID counter
            try:
                if hasattr(self._model, "predictor") and self._model.predictor is not None:
                    self._model.predictor = None
            except Exception:
                pass

    def track_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        extracted_frame_index: int,
        timestamp_seconds: float,
        frame_filename: str,
        conf_threshold: float = DEFAULT_TRACKING_CONF_THRESHOLD,
    ) -> Tuple[bool, List[TrackResult], str]:
        """Track visible people across frames, maintaining continuous track IDs.

        Args:
            frame: Numpy array representing image (RGB or BGR).
            frame_id: 0-indexed position in source video stream.
            extracted_frame_index: 1-indexed sequential frame number.
            timestamp_seconds: Exact elapsed time in seconds.
            frame_filename: Frame image filename.
            conf_threshold: Minimum confidence threshold (0.0 to 1.0).

        Returns:
            Tuple of (success: bool, tracks: List[TrackResult], message: str)
        """
        if self._model is None:
            ok, err = self.load_model()
            if not ok:
                return False, [], err

        is_valid, val_err = FramePreprocessor.validate_frame(frame)
        if not is_valid:
            return False, [], f"Invalid frame passed to tracker: {val_err}"

        if not (0.0 <= conf_threshold <= 1.0):
            return False, [], f"Confidence threshold must be between 0.0 and 1.0, got: {conf_threshold}"

        try:
            # Execute YOLO multi-object tracking with persistent state
            results = self._model.track(
                frame,
                persist=True,
                tracker=self.tracker_yaml,
                conf=conf_threshold,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )

            tracks: List[TrackResult] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])

                        # Ensure class is person and meets threshold
                        if cls_id == PERSON_CLASS_ID and conf >= conf_threshold:
                            coords = boxes.xyxy[i].tolist()
                            x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])

                            # Retrieve tracker ID (defaults to i+1 if unassigned)
                            if boxes.id is not None and len(boxes.id) > i:
                                track_id = int(boxes.id[i])
                            else:
                                track_id = i + 1

                            track_res = TrackResult(
                                frame_id=frame_id,
                                extracted_frame_index=extracted_frame_index,
                                timestamp_seconds=timestamp_seconds,
                                frame_filename=frame_filename,
                                track_id=track_id,
                                confidence=conf,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                            )
                            tracks.append(track_res)

                            # Record spatial trajectory point
                            cx, cy = track_res.center_x, track_res.center_y
                            if track_id not in self.trajectory_history:
                                self.trajectory_history[track_id] = []
                            self.trajectory_history[track_id].append((cx, cy))

                            # Keep history bounded
                            if len(self.trajectory_history[track_id]) > self.max_trajectory_points:
                                self.trajectory_history[track_id].pop(0)

            return True, tracks, ""
        except Exception as exc:
            return False, [], f"Inference error during person tracking: {str(exc)}"

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackResult],
        show_trajectories: bool = True,
    ) -> np.ndarray:
        """Render persistent colored bounding boxes, Track IDs, and centroid trajectories.

        Args:
            frame: Numpy array image (RGB format).
            tracks: List of TrackResult objects for the current frame.
            show_trajectories: Whether to render centroid movement history trails.

        Returns:
            Annotated frame array in RGB format.
        """
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw trajectory history trails first (so boxes appear on top)
        if show_trajectories:
            for track in tracks:
                t_id = track.track_id
                if t_id in self.trajectory_history and len(self.trajectory_history[t_id]) > 1:
                    pts = self.trajectory_history[t_id]
                    color = get_track_color(t_id)

                    for j in range(1, len(pts)):
                        pt1 = (int(round(pts[j - 1][0])), int(round(pts[j - 1][1])))
                        pt2 = (int(round(pts[j][0])), int(round(pts[j][1])))
                        # Dynamic line thickness fading forward
                        thickness = max(1, int(round(3.0 * (j / len(pts)))))
                        cv2.line(annotated, pt1, pt2, color, thickness, cv2.LINE_AA)

                    # Draw small circle at latest center point
                    latest_pt = (int(round(pts[-1][0])), int(round(pts[-1][1])))
                    cv2.circle(annotated, latest_pt, 4, color, -1, cv2.LINE_AA)

        # Draw bounding boxes and track labels
        for track in tracks:
            x1, y1 = int(round(track.x1)), int(round(track.y1))
            x2, y2 = int(round(track.x2)), int(round(track.y2))

            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            color = get_track_color(track.track_id)

            # Draw outer bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw center centroid dot
            cx, cy = int(round(track.center_x)), int(round(track.center_y))
            cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 2, color, -1, cv2.LINE_AA)

            # Format label with anonymous Track ID and confidence
            label = f"ID: {track.track_id} ({track.confidence:.2f})"

            # Calculate label background box size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Position label box above bounding box, or below if near top
            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = label_y1 + text_h + baseline + 4
            label_x2 = min(w - 1, x1 + text_w + 8)

            # Draw solid background badge
            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)

            # Contrast text color (white on dark/colored badge)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y2 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return annotated


def run_tracking_on_frames(
    video_id: str,
    frames_df: pd.DataFrame,
    tracker: Optional[PersonTracker] = None,
    conf_threshold: float = DEFAULT_TRACKING_CONF_THRESHOLD,
    tracker_type: str = DEFAULT_TRACKER,
    frame_selection_mode: str = "all",
    custom_sample_count: int = 15,
    show_trajectories: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[bool, Optional[TrackingSummary], Optional[pd.DataFrame], Dict[int, np.ndarray], str]:
    """Execute sequential multi-object tracking across classroom frames.

    Args:
        video_id: Unique video identifier.
        frames_df: DataFrame containing frame metadata from Feature 2.
        tracker: Optional PersonTracker instance.
        conf_threshold: Minimum detection confidence threshold (0.0 to 1.0).
        tracker_type: Algorithm name ("bytetrack" or "botsort").
        frame_selection_mode: "all", "sample", or "custom".
        custom_sample_count: Number of frames if custom mode is chosen.
        show_trajectories: Whether to draw spatial centroid movement trails.
        progress_callback: Progress reporting function callback(current, total, message).

    Returns:
        Tuple of (success, summary, tracks_df, annotated_frames_dict, status_message)
    """
    if frames_df is None or frames_df.empty:
        return False, None, None, {}, "No frame metadata available. Please run Frame Extraction first."

    # Validate required columns from Feature 2 metadata
    required_cols = {"frame_index", "extracted_frame_index", "timestamp_seconds", "frame_filename", "frame_path"}
    missing_cols = required_cols - set(frames_df.columns)
    if missing_cols:
        return False, None, None, {}, f"Frame metadata missing required columns: {missing_cols}"

    # Ensure chronological order
    sorted_frames_df = frames_df.sort_values(by="timestamp_seconds").reset_index(drop=True)

    # Frame subset selection
    total_available = len(sorted_frames_df)
    if frame_selection_mode == "first":
        subset_df = sorted_frames_df.iloc[:1].copy()
    elif frame_selection_mode == "sample":
        # First, middle, and last
        if total_available <= 3:
            subset_df = sorted_frames_df.copy()
        else:
            indices = [0, total_available // 2, total_available - 1]
            subset_df = sorted_frames_df.iloc[indices].copy()
    elif frame_selection_mode == "custom":
        k = max(1, min(custom_sample_count, total_available))
        indices = np.linspace(0, total_available - 1, k, dtype=int).tolist()
        subset_df = sorted_frames_df.iloc[indices].copy()
    else:  # "all"
        subset_df = sorted_frames_df.copy()

    subset_df = subset_df.reset_index(drop=True)
    num_frames = len(subset_df)

    if num_frames == 0:
        return False, None, None, {}, "No frames selected for tracking."

    # Initialize tracker if not provided
    if tracker is None:
        tracker = PersonTracker(tracker_type=tracker_type)

    tracker.reset()
    ok, load_err = tracker.load_model()
    if not ok:
        return False, None, None, {}, f"Could not load tracking model: {load_err}"

    all_tracks: List[TrackResult] = []
    annotated_frames: Dict[int, np.ndarray] = {}

    for idx, row in subset_df.iterrows():
        f_path = Path(str(row["frame_path"]))
        frame_idx = int(row["frame_index"])
        ext_frame_idx = int(row["extracted_frame_index"])
        timestamp = float(row["timestamp_seconds"])
        filename = str(row["frame_filename"])

        if progress_callback:
            progress_callback(
                int(idx) + 1,
                num_frames,
                f"Tracking people in frame {int(idx) + 1}/{num_frames} ({filename})...",
            )

        if not f_path.exists():
            continue

        # Load frame
        img_bgr = cv2.imread(str(f_path))
        if img_bgr is None or img_bgr.size == 0:
            continue

        # Convert to RGB for visualization integrity
        img_rgb = FramePreprocessor.bgr_to_rgb(img_bgr)

        # Track people in current frame
        success, tracks, _ = tracker.track_frame(
            frame=img_rgb,
            frame_id=frame_idx,
            extracted_frame_index=ext_frame_idx,
            timestamp_seconds=timestamp,
            frame_filename=filename,
            conf_threshold=conf_threshold,
        )

        if success:
            all_tracks.extend(tracks)
            annotated = tracker.draw_tracks(img_rgb, tracks, show_trajectories=show_trajectories)
            annotated_frames[ext_frame_idx] = annotated
        else:
            annotated_frames[ext_frame_idx] = img_rgb

    # Compile structured results DataFrame
    if all_tracks:
        records = [
            {
                "video_id": video_id,
                **t.to_dict(),
            }
            for t in all_tracks
        ]
        tracks_df = pd.DataFrame(records)
    else:
        # Create empty DataFrame with standardized schema
        tracks_df = pd.DataFrame(
            columns=[
                "video_id",
                "frame_id",
                "extracted_frame_index",
                "timestamp_seconds",
                "frame_filename",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
            ]
        )

    # Calculate metrics
    unique_tracks = int(tracks_df["track_id"].nunique()) if not tracks_df.empty else 0
    total_detections = len(tracks_df)
    avg_active = round(total_detections / max(num_frames, 1), 2)

    # Track length statistics
    if not tracks_df.empty:
        track_lengths = tracks_df.groupby("track_id").size()
        min_len = int(track_lengths.min())
        max_len = int(track_lengths.max())

        track_durations = tracks_df.groupby("track_id")["timestamp_seconds"].agg(lambda s: s.max() - s.min())
        longest_duration = round(float(track_durations.max()), 2)
    else:
        min_len = 0
        max_len = 0
        longest_duration = 0.0

    # Save to CSV
    output_dir = PROCESSED_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv_path = output_dir / TRACKS_CSV_FILENAME

    try:
        tracks_df.to_csv(tracks_csv_path, index=False)
    except Exception as exc:
        return False, None, tracks_df, annotated_frames, f"Failed to save tracks CSV: {str(exc)}"

    summary = TrackingSummary(
        video_id=video_id,
        tracker_name=tracker.tracker_type,
        confidence_threshold=conf_threshold,
        frames_processed=num_frames,
        unique_tracks=unique_tracks,
        avg_active_tracks_per_frame=avg_active,
        longest_track_duration_seconds=longest_duration,
        longest_track_frames=max_len,
        min_track_length=min_len,
        max_track_length=max_len,
        tracks_csv_path=tracks_csv_path,
        status="Completed",
    )

    return True, summary, tracks_df, annotated_frames, f"Successfully tracked {unique_tracks} unique people across {num_frames} frames."
