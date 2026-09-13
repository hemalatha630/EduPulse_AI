"""EduPulse AI - Temporal Learning-Engagement Profiling from Classroom Videos.

Feature 1: Project Setup + Classroom Video Input & Metadata Profiling.
Feature 2: Frame Extraction & Preprocessing with Temporal Metadata Preservation.
Feature 3: Student / Person Detection using Pretrained YOLO Object Detection.
"""

import math
from pathlib import Path
import sys

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.behaviour.behaviour_classifier import (
    BehaviourClassifier,
    BehaviourPrediction,
    BehaviourSummary,
    run_behaviour_recognition_on_tracks,
)
from src.behaviour.behaviour_labels import (
    ALL_BEHAVIOUR_CLASSES,
    BEHAVIOUR_DESCRIPTIONS,
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
    get_behaviour_description,
    get_behaviour_hex,
    get_behaviour_rgb,
)
from src.cnn import (
    CNNFeatureExtractor,
    compute_pca_2d,
    create_pca_scatter_figure,
    run_cnn_feature_extraction,
)
from src.config import (
    BEHAVIOUR_COLORS,
    BEHAVIOURS_CSV_FILENAME,
    CNN_FEATURE_DIM,
    CNN_FEATURES_NPY_FILENAME,
    CNN_METADATA_CSV_FILENAME,
    DEFAULT_BEHAVIOUR_CONF_THRESHOLD,
    DEFAULT_CNN_BATCH_SIZE,
    DEFAULT_CNN_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_FRAME_GAP,
    DEFAULT_SAMPLING_INTERVAL,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEQUENCE_STRIDE,
    DEFAULT_TRACKER,
    DEFAULT_TRACKING_CONF_THRESHOLD,
    DEFAULT_YOLO_MODEL,
    EXCLUDED_INTERNAL_STATES,
    FRAMES_DIR,
    PROCESSED_DIR,
    PROJECT_TITLE,
    SUPPORTED_CNN_MODELS,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_TRACKERS,
    TARGET_OBSERVABLE_BEHAVIOURS,
    TEMPORAL_SEQUENCES_METADATA_FILENAME,
    TEMPORAL_SEQUENCES_NPY_FILENAME,
    TRACKS_CSV_FILENAME,
    UNKNOWN_BEHAVIOUR,
    VIDEOS_DIR,
    ensure_directories,
)
from src.temporal import (
    ClassroomSequenceDataset,
    SequenceSummary,
    TemporalSequenceGenerator,
    create_sequence_timeline_figure,
    create_track_coverage_figure,
    run_temporal_sequence_creation,
    sequences_to_tensor,
)
from src.detection.detector import (
    DetectionResult,
    DetectionSummary,
    YOLOPersonDetector,
    run_detection_on_frames,
)
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.tracking.tracker import (
    PersonTracker,
    TrackResult,
    TrackingSummary,
    get_track_color,
    run_tracking_on_frames,
)
from src.video.frame_extractor import (
    ExtractionConfig,
    ExtractionSummary,
    calculate_sampling_info,
    derive_video_id,
    extract_video_frames,
)
from src.video.video_utils import (
    VideoMetadata,
    extract_video_metadata,
    save_uploaded_video,
    validate_file_extension,
)

# Page configuration
st.set_page_config(
    page_title="EduPulse AI | Classroom Video Input, Frames, Tracking, Behaviour & Temporal Sequences",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure data directories exist
ensure_directories()


@st.cache_resource
def get_yolo_detector(model_name: str = DEFAULT_YOLO_MODEL) -> YOLOPersonDetector:
    """Cache and return the loaded YOLO person detector instance."""
    detector = YOLOPersonDetector(model_name=model_name)
    detector.load_model()
    return detector


@st.cache_resource
def get_person_tracker(
    model_name: str = DEFAULT_YOLO_MODEL, tracker_type: str = DEFAULT_TRACKER
) -> PersonTracker:
    """Cache and return the loaded PersonTracker instance."""
    tracker = PersonTracker(model_name=model_name, tracker_type=tracker_type)
    tracker.load_model()
    return tracker


@st.cache_resource
def get_behaviour_classifier() -> BehaviourClassifier:
    """Cache and return the loaded BehaviourClassifier instance."""
    classifier = BehaviourClassifier()
    return classifier


@st.cache_resource
def get_cnn_feature_extractor(model_name: str = DEFAULT_CNN_MODEL) -> CNNFeatureExtractor:
    """Cache and return the loaded CNN visual feature extractor instance."""
    extractor = CNNFeatureExtractor(model_name=model_name)
    return extractor


def render_sidebar():
    """Render educational research context and pipeline information in sidebar."""
    with st.sidebar:
        st.title("🎓 EduPulse AI")
        st.caption("Temporal Classroom Learning-Engagement Profiling")

        st.markdown("---")
        st.subheader("📋 Research Scope")
        st.write(
            "This project analyzes classroom video to profile **observable learning-related behaviours** over time."
        )

        with st.expander("🎯 Target Observable Behaviours", expanded=True):
            for idx, behaviour in enumerate(TARGET_OBSERVABLE_BEHAVIOURS, 1):
                st.markdown(f"**{idx}.** {behaviour}")

        with st.expander("🛡️ Non-Detectable State Exclusions", expanded=False):
            st.info(
                "In accordance with rigorous computer vision and ethical pedagogical standards, "
                "this system does **not** detect internal mental or emotional states:"
            )
            for item in EXCLUDED_INTERNAL_STATES:
                st.markdown(f"• *{item}*")

        st.markdown("---")
        st.subheader("⚙️ Current Phase")
        st.success("✅ **Feature 1: Video Ingestion & Metadata**")
        st.success("✅ **Feature 2: Frame Extraction & Preprocessing**")
        st.success("✅ **Feature 3: Student / Person Detection**")
        st.success("✅ **Feature 4: Student / Person Tracking**")
        st.success("✅ **Feature 5: Observable Behaviour Recognition**")
        st.success("✅ **Feature 6: CNN Visual Feature Extraction**")
        st.success("🚀 **Feature 7: Temporal Sequence Creation**")
        st.caption("Next stages (Feature 8: Temporal Sequence Modelling RNN/LSTM) unlock in future milestones.")


def render_header():
    """Render main application header."""
    st.title(PROJECT_TITLE)
    st.markdown(
        "Upload a classroom recording (`.mp4`, `.avi`, `.mov`, `.mkv`), inspect stream properties, "
        "extract chronological preprocessed frames, detect students, and track people across time with persistent Track IDs."
    )
    st.divider()


def render_metadata_section(metadata: VideoMetadata):
    """Render technical metadata in structured cards and metrics."""
    st.subheader("📊 Video Information")

    # Primary metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Duration", value=metadata.duration_formatted)
    with col2:
        st.metric(label="Resolution", value=metadata.resolution)
    with col3:
        st.metric(label="Frame Rate", value=f"{metadata.fps} FPS")
    with col4:
        st.metric(label="Total Frames", value=f"{metadata.total_frames:,}")

    # Detailed specifications row
    st.markdown("##### Technical Specifications")
    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown(
            f"""
            - **Filename:** `{metadata.filename}`
            - **Container Format:** `{metadata.file_format}`
            - **File Size:** `{metadata.file_size_formatted}`
            """
        )
    with spec_col2:
        st.markdown(
            f"""
            - **Frame Width:** `{metadata.width} px`
            - **Frame Height:** `{metadata.height} px`
            - **Exact Duration:** `{metadata.duration_seconds} seconds`
            """
        )


def render_extraction_summary(summary: ExtractionSummary):
    """Display structured summary cards and metrics after frame extraction."""
    st.subheader("📋 Frame Extraction Summary")

    metric_c1, metric_c2, metric_c3, metric_c4 = st.columns(4)
    with metric_c1:
        st.metric("Original FPS", f"{summary.original_fps} FPS")
    with metric_c2:
        st.metric("Original Frames", f"{summary.original_total_frames:,}")
    with metric_c3:
        st.metric("Video Duration", summary.duration_formatted)
    with metric_c4:
        st.metric("Extracted Frames", f"{summary.extracted_frames_count:,}")

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown(
            f"""
            - **Sampling Setting:** `{summary.sampling_description}`
            - **Effective Sampling Rate:** `~{summary.effective_fps} FPS`
            - **Status:** `{summary.status}`
            """
        )
    with summary_col2:
        st.markdown(
            f"""
            - **Frames Directory:** `{summary.frames_directory}`
            - **Metadata CSV:** `{summary.metadata_path}`
            - **Source Video ID:** `{summary.video_id}`
            """
        )


def render_frame_previews(df: pd.DataFrame):
    """Display representative sample of extracted frames (First, Middle, Last)."""
    st.subheader("🖼️ Preview Extracted Frames")
    st.caption("Representative sample: First extracted frame, middle frame, and last extracted frame.")

    total_extracted = len(df)
    if total_extracted == 0:
        st.warning("No extracted frames available to preview.")
        return

    first_idx = 0
    middle_idx = total_extracted // 2
    last_idx = total_extracted - 1

    preview_indices = [
        ("First Extracted Frame", first_idx),
        ("Middle Extracted Frame", middle_idx),
        ("Last Extracted Frame", last_idx),
    ]

    cols = st.columns(3)
    for i, (title, row_idx) in enumerate(preview_indices):
        row = df.iloc[row_idx]
        frame_path = Path(row["frame_path"])
        frame_num = int(row["extracted_frame_index"])
        orig_frame_num = int(row["frame_index"])
        timestamp = float(row["timestamp_seconds"])

        with cols[i]:
            st.markdown(f"**{title}**")
            if frame_path.exists():
                # Load frame and ensure RGB conversion for proper display
                read_ok, bgr_img, _ = FramePreprocessor.load_frame(frame_path)
                if read_ok and bgr_img is not None:
                    rgb_img = FramePreprocessor.bgr_to_rgb(bgr_img)
                    st.image(
                        rgb_img,
                        caption=f"Frame #{frame_num:03d} (Orig #{orig_frame_num}) | {timestamp:.2f}s",
                        use_container_width=True,
                    )
                else:
                    st.error("Failed to load frame for preview.")
            else:
                st.error(f"Image not found: {frame_path.name}")


def render_metadata_table(df: pd.DataFrame, video_id: str):
    """Render interactive metadata table and CSV download button."""
    st.subheader("📄 Temporal Frame Metadata")
    st.caption("Chronological record mapping every extracted frame to its video timestamp and disk path.")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Frame Metadata CSV",
        data=csv_data,
        file_name=f"{video_id}_frame_metadata.csv",
        mime="text/csv",
        help="Download the complete temporal index linking frame files to timestamps.",
    )


def render_frame_extraction_section(video_path: Path, metadata: VideoMetadata):
    """Render Feature 2: Frame extraction controls, progress, summary, and preview."""
    st.subheader("🎞️ Feature 2: Frame Extraction & Preprocessing")
    st.markdown(
        "Extract chronological frames from the uploaded video stream with configurable temporal sampling. "
        "Extracted frames are preprocessed, indexed with exact timestamps, and persisted to `data/frames/`."
    )

    # Configuration layout
    ctrl_col1, ctrl_col2 = st.columns([1.2, 1.0], gap="medium")

    with ctrl_col1:
        st.markdown("##### ⏱️ Frame Sampling Setting")
        sampling_options = [
            "Every 5th frame (~6 FPS, Recommended for engagement analysis)",
            "Every frame (1:1 full extraction)",
            "Every 10th frame (~3 FPS)",
            "Every 15th frame (~2 FPS)",
            "Every 30th frame (~1 FPS)",
            "Custom sampling interval...",
        ]
        selected_option = st.selectbox(
            label="Select sampling frequency:",
            options=sampling_options,
            index=0,
            help="Subsampling significantly reduces storage and compute overhead while preserving temporal dynamics.",
        )

        if "Custom" in selected_option:
            sampling_interval = st.number_input(
                "Enter custom frame interval (N):",
                min_value=1,
                max_value=max(metadata.total_frames, 1),
                value=DEFAULT_SAMPLING_INTERVAL,
                step=1,
            )
        elif "Every frame" in selected_option:
            sampling_interval = 1
        elif "10th" in selected_option:
            sampling_interval = 10
        elif "15th" in selected_option:
            sampling_interval = 15
        elif "30th" in selected_option:
            sampling_interval = 30
        else:
            sampling_interval = 5

        effective_fps, sampling_desc = calculate_sampling_info(metadata.fps, sampling_interval)
        est_extracted = math.ceil(metadata.total_frames / sampling_interval) if metadata.total_frames > 0 else 0

        # Dynamic calculated metrics banner
        st.info(
            f"**Original FPS:** `{metadata.fps}`  |  "
            f"**Sampling:** `{sampling_desc}`  |  "
            f"**Effective Sampling Rate:** `~{effective_fps} FPS`  |  "
            f"**Estimated Frames:** `~{est_extracted:,}`"
        )

    with ctrl_col2:
        st.markdown("##### 🛠️ Basic Preprocessing Options")
        resolution_option = st.selectbox(
            label="Resolution handling:",
            options=[
                f"Original resolution ({metadata.resolution})",
                "1280 x 720 (720p HD)",
                "640 x 360 (360p Fast)",
            ],
            index=0,
            help="Keeps full pixel fidelity by default. Downscaling can be enabled if storage is constrained.",
        )

        target_res = None
        if "720p" in resolution_option:
            target_res = (1280, 720)
        elif "360p" in resolution_option:
            target_res = (640, 360)

        force_reextract = st.checkbox(
            "Force re-extraction (reprocess even if frames already exist)",
            value=False,
            help="By default, existing valid extractions are reused to avoid duplicate processing.",
        )

    # Session state key for results
    video_id = derive_video_id(video_path)
    state_key = f"extraction_{video_id}"

    # Extraction Trigger Button
    extract_clicked = st.button(
        "🎬 Extract Frames",
        type="primary",
        use_container_width=True,
        key="btn_extract_frames",
    )

    if extract_clicked:
        cfg = ExtractionConfig(
            sampling_interval=sampling_interval,
            target_resolution=target_res,
            jpeg_quality=95,
            force_reextract=force_reextract,
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(current: int, total: int, msg: str):
            fraction = min(max(current / max(total, 1), 0.0), 1.0)
            progress_bar.progress(fraction)
            status_text.caption(f"⏳ {msg} ({int(fraction * 100)}%)")

        with st.spinner("Extracting and preprocessing video frames..."):
            success, summary, df, msg = extract_video_frames(
                video_path=video_path,
                config=cfg,
                progress_callback=on_progress,
            )

        progress_bar.empty()
        status_text.empty()

        if success and summary and df is not None:
            st.session_state[state_key] = {"summary": summary, "df": df}
            if summary.is_cached:
                st.info(f"ℹ️ {msg}")
            else:
                st.success(f"✅ {msg}")
        else:
            st.error(f"❌ Extraction failed: {msg}")

    # Render results if available in session state
    if state_key in st.session_state:
        res = st.session_state[state_key]
        summary: ExtractionSummary = res["summary"]
        df: pd.DataFrame = res["df"]

        st.markdown("---")
        render_extraction_summary(summary)

        st.markdown("---")
        render_frame_previews(df)

        st.markdown("---")
        render_metadata_table(df, summary.video_id)


def render_detection_summary(summary: DetectionSummary):
    """Display structured summary cards and metrics after person detection."""
    st.subheader("📋 Detection Summary")

    metric_c1, metric_c2, metric_c3, metric_c4, metric_c5 = st.columns(5)
    with metric_c1:
        st.metric("Frames Processed", f"{summary.frames_processed:,}")
    with metric_c2:
        st.metric("Total Detections", f"{summary.total_person_detections:,}")
    with metric_c3:
        st.metric("Avg Detections/Frame", f"{summary.avg_detections_per_frame}")
    with metric_c4:
        st.metric("Min Detections", f"{summary.min_detections}")
    with metric_c5:
        st.metric("Max Detections", f"{summary.max_detections}")

    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown(
            f"""
            - **Detection Model:** `{summary.model_name}` (Pretrained COCO)
            - **Confidence Threshold:** `{summary.confidence_threshold:.2f}`
            """
        )
    with spec_col2:
        st.markdown(
            f"""
            - **Target Class:** `person` (Class ID 0)
            - **Detections CSV:** `{summary.detections_csv_path}`
            """
        )


def render_detection_section(video_path: Path):
    """Render Feature 3: Student / Person Detection controls and visualizations."""
    st.subheader("👥 Feature 3: Student / Person Detection")
    st.markdown(
        "Detect visible people in classroom frames using a lightweight pretrained YOLO object detector. "
        "Detections are computed independently per frame with bounding boxes and confidence scores. "
        "*(Note: Persistent student tracking IDs are not assigned; tracking will be introduced in Feature 4).* "
    )

    video_id = derive_video_id(video_path)
    metadata_csv_path = PROCESSED_DIR / video_id / "frame_metadata.csv"
    frames_dir = FRAMES_DIR / video_id

    # Verify extracted frames exist
    if not metadata_csv_path.exists() or not frames_dir.exists():
        st.info("👉 Please complete **Feature 2 (Frame Extraction)** above to generate frames for person detection.")
        return

    try:
        frames_df = pd.read_csv(metadata_csv_path)
        if frames_df.empty:
            st.info("👉 No extracted frames found in metadata. Please run frame extraction first.")
            return
    except Exception as exc:
        st.error(f"Error reading frame metadata: {str(exc)}")
        return

    det_ctrl_col1, det_ctrl_col2 = st.columns([1.1, 1.0], gap="medium")

    with det_ctrl_col1:
        st.markdown("##### 🎯 Detection Confidence Threshold")
        conf_threshold = st.slider(
            label="Minimum Confidence Threshold:",
            min_value=0.10,
            max_value=1.00,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Filters out bounding box detections with confidence scores below this threshold.",
            key="slider_conf_threshold",
        )
        st.caption(f"Currently filtering detections with confidence $\\ge {conf_threshold:.2f}$.")

    with det_ctrl_col2:
        st.markdown("##### 🖼️ Frame Selection")
        selection_mode_label = st.selectbox(
            label="Frames to process:",
            options=[
                "Sample representative frames (First, Middle, Last)",
                "First frame only",
                "Custom sample count (N evenly spaced frames)",
                "All extracted frames (May take longer)",
            ],
            index=0,
            help="Choose how many extracted frames to run person detection on.",
            key="sb_frame_selection",
        )

        custom_count = 5
        if "Custom" in selection_mode_label:
            custom_count = st.number_input(
                "Number of sample frames (N):",
                min_value=1,
                max_value=max(len(frames_df), 1),
                value=min(5, len(frames_df)),
                step=1,
            )

    # Map selection label to mode
    if "First frame" in selection_mode_label:
        selection_mode = "first"
    elif "All" in selection_mode_label:
        selection_mode = "all"
    elif "Custom" in selection_mode_label:
        selection_mode = "custom"
    else:
        selection_mode = "sample"

    det_state_key = f"detection_{video_id}"

    detect_clicked = st.button(
        "🔍 Detect People in Frames",
        type="primary",
        use_container_width=True,
        key="btn_detect_persons",
    )

    if detect_clicked:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_det_progress(current: int, total: int, msg: str):
            fraction = min(max(current / max(total, 1), 0.0), 1.0)
            progress_bar.progress(fraction)
            status_text.caption(f"⏳ {msg} ({int(fraction * 100)}%)")

        with st.spinner("Initializing YOLO model and running person detection..."):
            try:
                detector = get_yolo_detector(DEFAULT_YOLO_MODEL)
                success, summary, det_df, annotated_frames, msg = run_detection_on_frames(
                    video_id=video_id,
                    frames_df=frames_df,
                    detector=detector,
                    conf_threshold=conf_threshold,
                    frame_selection_mode=selection_mode,
                    custom_sample_count=int(custom_count),
                    progress_callback=on_det_progress,
                )
            except Exception as exc:
                success = False
                summary = None
                det_df = None
                annotated_frames = {}
                msg = f"Unexpected detection error: {str(exc)}"

        progress_bar.empty()
        status_text.empty()

        if success and summary and det_df is not None:
            st.session_state[det_state_key] = {
                "summary": summary,
                "df": det_df,
                "annotated": annotated_frames,
            }
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ Detection failed: {msg}")

    # Render detection results if present in session state
    if det_state_key in st.session_state:
        det_res = st.session_state[det_state_key]
        summary: DetectionSummary = det_res["summary"]
        det_df: pd.DataFrame = det_res["df"]
        annotated_frames: Dict[int, np.ndarray] = det_res["annotated"]

        st.markdown("---")
        render_detection_summary(summary)

        st.markdown("---")
        st.subheader("🖼️ Detection Visualizer")

        if not annotated_frames:
            st.warning("⚠️ No people were detected in the selected frame(s) using the current confidence threshold.")
        else:
            available_frame_indices = sorted(annotated_frames.keys())
            if len(available_frame_indices) == 1:
                selected_frame_idx = available_frame_indices[0]
            else:
                selected_frame_idx = st.select_slider(
                    "Select frame to inspect bounding box detections:",
                    options=available_frame_indices,
                    value=available_frame_indices[0],
                    format_func=lambda idx: f"Frame #{idx:03d}",
                )

            annotated_img = annotated_frames[selected_frame_idx]
            frame_dets = det_df[det_df["frame_id"] == selected_frame_idx]
            people_count = len(frame_dets)

            st.markdown(f"**Showing Frame #{selected_frame_idx:03d}** — People Detected: `{people_count}`")
            st.image(
                annotated_img,
                caption=f"Frame #{selected_frame_idx:03d} | Detected People: {people_count} | Conf >= {summary.confidence_threshold:.2f}",
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("📄 Detections Dataset")
        st.caption("Structured bounding box coordinates and confidence scores per detected person.")

        st.dataframe(
            det_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = det_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Detections CSV",
            data=csv_data,
            file_name=f"{video_id}_detections.csv",
            mime="text/csv",
            help="Download bounding box coordinates and confidence metrics for all detections.",
        )


def render_tracking_summary(summary: TrackingSummary):
    """Display structured summary cards and metrics after multi-object tracking."""
    st.subheader("📋 Tracking Summary")

    metric_c1, metric_c2, metric_c3, metric_c4 = st.columns(4)
    with metric_c1:
        st.metric("Frames Processed", f"{summary.frames_processed:,}")
    with metric_c2:
        st.metric("Unique Tracks", f"{summary.unique_tracks:,}")
    with metric_c3:
        st.metric("Avg Active Tracks/Frame", f"{summary.avg_active_tracks_per_frame}")
    with metric_c4:
        st.metric(
            "Longest Track Duration",
            f"{summary.longest_track_duration_seconds}s",
            help=f"Continuous tracking over {summary.longest_track_frames} frames",
        )

    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown(
            f"""
            - **Tracker Algorithm:** `{summary.tracker_name.upper()}`
            - **Confidence Threshold:** `{summary.confidence_threshold:.2f}`
            """
        )
    with spec_col2:
        st.markdown(
            f"""
            - **Track Length Range:** `{summary.min_track_length}` min / `{summary.max_track_length}` max frames
            - **Tracks CSV:** `{summary.tracks_csv_path}`
            """
        )


def render_tracking_section(video_path: Path):
    """Render Feature 4: Student / Person Tracking controls and visualizations."""
    st.subheader("🧭 Feature 4: Student / Person Tracking")
    st.markdown(
        "Connect per-frame person detections across consecutive video frames into persistent, anonymous **Track IDs** "
        "using multi-object tracking (ByteTrack or BoT-SORT). Track IDs allow the system to trace individual student spatial positions across time."
    )

    st.info(
        "🛡️ **Important Ethical Scope Notice:** Track IDs (e.g. `ID: 1`, `ID: 2`) are temporary computational identifiers "
        "assigned exclusively to maintain spatial continuity across frames. They **do not** represent real student identities, "
        "names, roll numbers, or personal profiles. Facial recognition is strictly excluded."
    )

    video_id = derive_video_id(video_path)
    metadata_csv_path = PROCESSED_DIR / video_id / "frame_metadata.csv"
    frames_dir = FRAMES_DIR / video_id

    # Verify extracted frames exist
    if not metadata_csv_path.exists() or not frames_dir.exists():
        st.info("👉 Please complete **Feature 2 (Frame Extraction)** above to generate frames for person tracking.")
        return

    try:
        frames_df = pd.read_csv(metadata_csv_path)
        if frames_df.empty:
            st.info("👉 No extracted frames found in metadata. Please run frame extraction first.")
            return
    except Exception as exc:
        st.error(f"Error reading frame metadata: {str(exc)}")
        return

    track_ctrl_col1, track_ctrl_col2, track_ctrl_col3 = st.columns([1.0, 1.0, 1.2], gap="medium")

    with track_ctrl_col1:
        st.markdown("##### ⚙️ Tracker Algorithm")
        tracker_type = st.selectbox(
            label="Select tracking method:",
            options=SUPPORTED_TRACKERS,
            index=0,
            format_func=lambda x: "ByteTrack (Recommended - High Speed)" if x == "bytetrack" else "BoT-SORT (Camera Motion Compensation)",
            help="ByteTrack associates both high and low score detection boxes to maintain tracks through occlusion.",
            key="sb_tracker_type",
        )

    with track_ctrl_col2:
        st.markdown("##### 🎯 Confidence Threshold")
        conf_threshold = st.slider(
            label="Minimum Tracking Confidence:",
            min_value=0.10,
            max_value=1.00,
            value=DEFAULT_TRACKING_CONF_THRESHOLD,
            step=0.05,
            help="Filters out bounding box detections with confidence scores below this threshold before tracking.",
            key="slider_track_conf",
        )
        st.caption(f"Tracking detections with confidence $\\ge {conf_threshold:.2f}$.")

    with track_ctrl_col3:
        st.markdown("##### 🎞️ Frames to Track")
        seq_mode_label = st.selectbox(
            label="Sequence range:",
            options=[
                "All extracted frames (Continuous sequence)",
                "First 15 frames (Quick preview)",
                "First 30 frames (Extended preview)",
                "Custom frame count (N frames)",
            ],
            index=0,
            help="Tracking requires consecutive chronological frames for consistent ID continuity.",
            key="sb_track_seq",
        )

        custom_track_count = 15
        if "Custom" in seq_mode_label:
            custom_track_count = st.number_input(
                "Number of consecutive frames (N):",
                min_value=2,
                max_value=max(len(frames_df), 2),
                value=min(15, len(frames_df)),
                step=1,
                key="num_input_track_count",
            )

    show_trajectories = st.checkbox(
        "Draw spatial centroid trajectory paths (Movement history trails)",
        value=True,
        help="Renders historical centroid movement lines behind each tracked person across consecutive frames.",
        key="cb_show_trajectories",
    )

    # Map sequence label to mode
    if "First 15" in seq_mode_label:
        selection_mode = "custom"
        custom_track_count = 15
    elif "First 30" in seq_mode_label:
        selection_mode = "custom"
        custom_track_count = 30
    elif "Custom" in seq_mode_label:
        selection_mode = "custom"
    else:
        selection_mode = "all"

    tracking_state_key = f"tracking_{video_id}"

    track_clicked = st.button(
        "🎯 Track People Across Consecutive Frames",
        type="primary",
        use_container_width=True,
        key="btn_track_persons",
    )

    if track_clicked:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_track_progress(current: int, total: int, msg: str):
            fraction = min(max(current / max(total, 1), 0.0), 1.0)
            progress_bar.progress(fraction)
            status_text.caption(f"⏳ {msg} ({int(fraction * 100)}%)")

        with st.spinner(f"Running {tracker_type.upper()} multi-object tracking across consecutive frames..."):
            try:
                tracker = get_person_tracker(DEFAULT_YOLO_MODEL, tracker_type)
                success, summary, tracks_df, annotated_frames, msg = run_tracking_on_frames(
                    video_id=video_id,
                    frames_df=frames_df,
                    tracker=tracker,
                    conf_threshold=conf_threshold,
                    tracker_type=tracker_type,
                    frame_selection_mode=selection_mode,
                    custom_sample_count=int(custom_track_count),
                    show_trajectories=show_trajectories,
                    progress_callback=on_track_progress,
                )
            except Exception as exc:
                success = False
                summary = None
                tracks_df = None
                annotated_frames = {}
                msg = f"Unexpected tracking error: {str(exc)}"

        progress_bar.empty()
        status_text.empty()

        if success and summary and tracks_df is not None:
            st.session_state[tracking_state_key] = {
                "summary": summary,
                "df": tracks_df,
                "annotated": annotated_frames,
            }
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ Tracking failed: {msg}")

    # Render tracking results if present in session state
    if tracking_state_key in st.session_state:
        res = st.session_state[tracking_state_key]
        summary: TrackingSummary = res["summary"]
        tracks_df: pd.DataFrame = res["df"]
        annotated_frames: Dict[int, np.ndarray] = res["annotated"]

        st.markdown("---")
        render_tracking_summary(summary)

        st.markdown("---")
        st.subheader("🖼️ Sequential Tracking Visualizer")

        if not annotated_frames:
            st.warning("⚠️ No tracked people found in the selected frames with the current settings.")
        else:
            available_frame_indices = sorted(annotated_frames.keys())

            selected_frame_idx = st.select_slider(
                "Select frame to inspect tracking continuity:",
                options=available_frame_indices,
                value=available_frame_indices[0],
                format_func=lambda idx: f"Frame #{idx:03d}",
                key="slider_track_frame",
            )

            # Tabbed inspection view: Single frame vs Consecutive comparison
            tab_single, tab_compare, tab_trajectory = st.tabs([
                "📸 Single Frame View",
                "🔄 Consecutive Frame Comparison (N-1 vs N)",
                "📈 Spatial Centroid Trajectory Map",
            ])

            with tab_single:
                annotated_img = annotated_frames[selected_frame_idx]
                curr_tracks = tracks_df[tracks_df["extracted_frame_index"] == selected_frame_idx]
                active_count = len(curr_tracks)

                st.markdown(
                    f"**Frame #{selected_frame_idx:03d}** — Active Tracked People: `{active_count}` | "
                    f"Track IDs present: `{sorted(curr_tracks['track_id'].tolist()) if active_count > 0 else 'None'}`"
                )
                st.image(
                    annotated_img,
                    caption=f"Frame #{selected_frame_idx:03d} | Active Tracks: {active_count} | Persistent IDs & Centroid Paths",
                    use_container_width=True,
                )

                if not curr_tracks.empty:
                    with st.expander(f"📋 Active Tracks Data in Frame #{selected_frame_idx:03d}", expanded=False):
                        display_cols = ["track_id", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
                        st.dataframe(
                            curr_tracks[display_cols].sort_values(by="track_id"),
                            use_container_width=True,
                            hide_index=True,
                        )

            with tab_compare:
                # Find preceding frame in sequence
                curr_pos = available_frame_indices.index(selected_frame_idx)
                if curr_pos == 0:
                    st.info("ℹ️ Currently viewing the first frame. Move the slider to Frame 2 or higher to compare consecutive frames.")
                    st.image(
                        annotated_frames[selected_frame_idx],
                        caption=f"Initial Frame #{selected_frame_idx:03d}",
                        use_container_width=True,
                    )
                else:
                    prev_frame_idx = available_frame_indices[curr_pos - 1]
                    prev_tracks = tracks_df[tracks_df["extracted_frame_index"] == prev_frame_idx]
                    curr_tracks = tracks_df[tracks_df["extracted_frame_index"] == selected_frame_idx]

                    prev_ids = set(prev_tracks["track_id"])
                    curr_ids = set(curr_tracks["track_id"])
                    shared_ids = prev_ids.intersection(curr_ids)

                    st.markdown(
                        f"**Comparing Frame #{prev_frame_idx:03d} (Previous) with Frame #{selected_frame_idx:03d} (Current)** — "
                        f"Shared Track IDs: `{sorted(shared_ids)}` ({len(shared_ids)} persisting tracks)"
                    )

                    col_prev, col_curr = st.columns(2, gap="medium")
                    with col_prev:
                        st.markdown(f"##### Previous: Frame #{prev_frame_idx:03d}")
                        st.image(
                            annotated_frames[prev_frame_idx],
                            caption=f"Frame #{prev_frame_idx:03d} (Tracks: {len(prev_tracks)})",
                            use_container_width=True,
                        )
                    with col_curr:
                        st.markdown(f"##### Current: Frame #{selected_frame_idx:03d}")
                        st.image(
                            annotated_frames[selected_frame_idx],
                            caption=f"Frame #{selected_frame_idx:03d} (Tracks: {len(curr_tracks)})",
                            use_container_width=True,
                        )

            with tab_trajectory:
                st.markdown("##### 📍 2D Spatial Centroid Movement Map")
                st.caption(
                    "Traces bounding box center points $(center\\_x, center\\_y)$ across processed video frames. "
                    "*(Note: Represents spatial physical movement in camera view only; not an engagement or cognitive metric)*."
                )

                if not tracks_df.empty and tracks_df["track_id"].nunique() > 0:
                    try:
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0e1117")
                        ax.set_facecolor("#161b22")

                        # Group by track_id and plot centroid path
                        from src.tracking.tracker import get_track_color

                        for t_id, group in tracks_df.groupby("track_id"):
                            if len(group) > 1:
                                group_sorted = group.sort_values(by="timestamp_seconds")
                                xs = group_sorted["center_x"].values
                                ys = group_sorted["center_y"].values
                                r, g, b = get_track_color(int(t_id))
                                hex_col = f"#{r:02x}{g:02x}{b:02x}"
                                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, color=hex_col, label=f"Track {t_id}")
                                ax.text(xs[0], ys[0], f"#{t_id}", fontsize=7, color="#ffffff", alpha=0.7)

                        ax.set_title("Track Centroid Trajectories Across Frames", color="#e6edf3", fontsize=12)
                        ax.set_xlabel("Frame X Coordinate (pixels)", color="#8b949e")
                        ax.set_ylabel("Frame Y Coordinate (pixels)", color="#8b949e")
                        ax.invert_yaxis()  # Invert Y so top of image is at top of plot
                        ax.tick_params(colors="#8b949e")
                        for spine in ax.spines.values():
                            spine.set_color("#30363d")
                        ax.grid(True, linestyle="--", alpha=0.2, color="#8b949e")

                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as plot_err:
                        st.info(f"Trajectory plotting note: {plot_err}")
                else:
                    st.info("No multi-frame track trajectories to display.")

        st.markdown("---")
        st.subheader("📄 Tracks Dataset (`tracks.csv`)")
        st.caption("Complete temporal tracking records with bounding box coordinates and centroids.")

        st.dataframe(
            tracks_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = tracks_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Tracks CSV",
            data=csv_data,
            file_name=f"{video_id}_tracks.csv",
            mime="text/csv",
            help="Download structured tracking records with persistent Track IDs and centroid coordinates.",
            key="btn_download_tracks_csv",
        )

        with st.expander("⚠️ Tracking Limitations & Occlusion Handling in Classrooms", expanded=False):
            st.markdown(
                """
                - **Classroom Density & Occlusion:** When students sit in dense lecture rows or walk past each other, temporary occlusion can occur.
                - **ID Switches:** If a student is completely hidden by another person or a laptop screen for several frames, the tracker may re-detect them under a new Track ID upon reappearance.
                - **Re-identification:** ByteTrack maintains tracks by associating low-confidence boxes in ambiguous situations, significantly reducing false track terminations.
                - **Academic Integrity:** Track IDs describe tracking performance and visual continuity — they never imply student identities, attendance records, or mental states.
                """
            )


def render_behaviour_recognition_section(video_path: Path):
    """Render Section 5: Observable Behaviour Recognition."""
    st.subheader("🎯 Feature 5: Observable Behaviour Recognition")
    st.info(
        "🛡️ **Strict Research Scope Notice:** This module identifies **strictly observable learning-related behaviours** "
        "from tracked person crops (posture, head orientation, reading/writing actions, peer proximity). "
        "It **does NOT** infer internal cognitive or emotional states (e.g. boredom, motivation, intelligence, attention, or comprehension)."
    )

    video_id = derive_video_id(video_path)
    metadata_csv_path = PROCESSED_DIR / video_id / "frame_metadata.csv"
    tracks_csv_path = PROCESSED_DIR / video_id / TRACKS_CSV_FILENAME
    frames_dir = FRAMES_DIR / video_id

    # Verify prerequisites: frames and tracks
    if not metadata_csv_path.exists() or not frames_dir.exists():
        st.info("👉 Please complete **Feature 2 (Frame Extraction)** above to extract video frames.")
        return

    if not tracks_csv_path.exists():
        st.info("👉 Please complete **Feature 4 (Student Tracking)** above to generate tracked person bounding boxes across consecutive frames.")
        return

    try:
        frames_df = pd.read_csv(metadata_csv_path)
        tracks_df = pd.read_csv(tracks_csv_path)
        if frames_df.empty:
            st.info("👉 No extracted frames found in metadata.")
            return
        if tracks_df.empty:
            st.info("👉 No tracking records found in tracks.csv.")
            return
    except Exception as exc:
        st.error(f"Error reading metadata or tracking data: {str(exc)}")
        return

    classifier = get_behaviour_classifier()

    st.success(f"🔬 **Classifier Mode:** {classifier.mode_name}")
    st.caption(
        "ℹ️ **Research Integrity Notice:** No proprietary labelled classroom behaviour dataset is currently loaded. "
        "The system operates in a transparent **Prototype / Baseline Heuristic** mode using observable visual cues "
        "(head orientation, aspect ratio, desk/hand region gradients, and proximal peer orientation). "
        "The architecture is fully modular, allowing trained PyTorch weights to be loaded seamlessly once a labelled dataset is acquired."
    )

    beh_ctrl_col1, beh_ctrl_col2 = st.columns([1.0, 1.2], gap="medium")

    with beh_ctrl_col1:
        st.markdown("##### 🎯 Classification Confidence Threshold")
        beh_conf_threshold = st.slider(
            label="Minimum Confidence Threshold:",
            min_value=0.10,
            max_value=1.00,
            value=DEFAULT_BEHAVIOUR_CONF_THRESHOLD,
            step=0.05,
            help="Predictions with confidence below this threshold are marked as 'Unknown / Uncertain' to preserve scientific rigor.",
            key="slider_beh_conf",
        )
        st.caption(f"Crops with classifier confidence $< {beh_conf_threshold:.2f}$ will be classified as *Unknown / Uncertain*.")

    with beh_ctrl_col2:
        st.markdown("##### 🎞️ Frames to Classify")
        beh_seq_label = st.selectbox(
            label="Frame range for behaviour recognition:",
            options=[
                "All extracted frames (Complete sequence)",
                "First 10 frames (Quick preview)",
                "First 18 frames (Standard sample)",
                "Custom frame count",
            ],
            index=0,
            help="Classify observable behaviours for tracked people across selected chronological frames.",
            key="sb_beh_seq",
        )

        custom_beh_count = 10
        if "Custom" in beh_seq_label:
            custom_beh_count = st.number_input(
                "Number of frames to process:",
                min_value=1,
                max_value=len(frames_df),
                value=min(10, len(frames_df)),
                step=1,
                key="num_input_beh_count",
            )

    # Determine frame selection mode
    if "All" in beh_seq_label:
        sel_mode = "all"
        first_n_val = len(frames_df)
    elif "First 10" in beh_seq_label:
        sel_mode = "first_n"
        first_n_val = 10
    elif "First 18" in beh_seq_label:
        sel_mode = "first_n"
        first_n_val = 18
    else:
        sel_mode = "first_n"
        first_n_val = int(custom_beh_count)

    run_beh_btn = st.button(
        "🎯 Recognise Observable Behaviours Across Tracks",
        type="primary",
        help="Crop each tracked person, evaluate observable visual cues, and assign one of the 6 defined behaviour categories.",
        key="btn_run_behaviour_recognition",
    )

    session_summary_key = f"beh_summary_{video_id}"
    session_df_key = f"beh_df_{video_id}"
    session_frames_key = f"beh_frames_{video_id}"

    if run_beh_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(min(1.0, max(0.0, pct)))
            status_text.text(msg)

        with st.spinner("Processing tracked person crops and classifying observable behaviours..."):
            success, summary, b_df, annotated_dict, msg = run_behaviour_recognition_on_tracks(
                video_id=video_id,
                frames_df=frames_df,
                tracks_df=tracks_df,
                classifier=classifier,
                conf_threshold=beh_conf_threshold,
                frame_selection_mode=sel_mode,
                first_n=first_n_val,
                progress_callback=update_progress,
            )

        progress_bar.empty()
        status_text.empty()

        if success and summary and b_df is not None:
            st.session_state[session_summary_key] = summary
            st.session_state[session_df_key] = b_df
            st.session_state[session_frames_key] = annotated_dict
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ Behaviour recognition failed: {msg}")

    # Check for existing results in session_state or saved CSV
    summary = st.session_state.get(session_summary_key)
    beh_df = st.session_state.get(session_df_key)
    annotated_dict = st.session_state.get(session_frames_key, {})

    behaviours_csv_path = PROCESSED_DIR / video_id / BEHAVIOURS_CSV_FILENAME
    if beh_df is None and behaviours_csv_path.exists():
        try:
            beh_df = pd.read_csv(behaviours_csv_path)
            st.session_state[session_df_key] = beh_df
        except Exception:
            pass

    if beh_df is not None and not beh_df.empty:
        st.markdown("---")
        st.subheader("📋 Behaviour Recognition Summary")

        # Metric cards
        total_obs = len(beh_df)
        known_df = beh_df[beh_df["behaviour_class"] != CLASS_UNKNOWN]
        dominant_beh = known_df["behaviour_class"].mode()[0] if len(known_df) > 0 else "N/A"
        unique_tracks_cov = beh_df["track_id"].nunique()
        unknown_count = len(beh_df[beh_df["behaviour_class"] == CLASS_UNKNOWN])

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            st.metric("Total Observations", f"{total_obs:,}")
            st.caption("Frame-level person classifications")
        with b_c2:
            st.metric("Most Frequent Behaviour", dominant_beh)
            st.caption("Dominant observable category")
        with b_c3:
            st.metric("Unique Tracks Covered", f"{unique_tracks_cov}")
            st.caption("Persistent student tracks")
        with b_c4:
            st.metric("Unknown / Uncertain", f"{unknown_count}")
            st.caption(f"Below {beh_conf_threshold:.2f} conf or occluded")

        st.markdown("---")

        # Visualizations row: Distribution & Details
        chart_col, exp_col = st.columns([1.2, 1.0], gap="large")

        with chart_col:
            st.markdown("##### 📊 Observable Behaviour Distribution")
            st.caption("Distribution of observable behaviours across all frame-level person observations.")

            class_counts = beh_df["behaviour_class"].value_counts()
            chart_df = pd.DataFrame({
                "Behaviour Category": class_counts.index,
                "Observations": class_counts.values,
            })

            # Horizontal bar chart using matplotlib for exact class colors
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4.5))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")

                categories = list(reversed(chart_df["Behaviour Category"].tolist()))
                counts = list(reversed(chart_df["Observations"].tolist()))
                bar_colors = [get_behaviour_hex(c) for c in categories]

                bars = ax.barh(categories, counts, color=bar_colors, edgecolor="#30363d", height=0.6)
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(w)}",
                            ha="left", va="center", color="#e6edf3", fontsize=9, fontweight="bold")

                ax.set_xlabel("Observation Count (Frame-level)", color="#8b949e", fontsize=10)
                ax.tick_params(colors="#e6edf3", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color("#30363d")
                ax.grid(axis="x", linestyle="--", alpha=0.2, color="#8b949e")

                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                st.bar_chart(chart_df.set_index("Behaviour Category"))

        with exp_col:
            st.markdown("##### 🔬 Observable Evidence & Boundaries")
            for b_name in TARGET_BEHAVIOUR_CLASSES:
                b_desc = get_behaviour_description(b_name)
                with st.expander(f"📌 {b_name}", expanded=False):
                    st.markdown(f"**Visual Evidence:** {b_desc['observable_evidence']}")
                    st.caption(f"⚠️ *Boundary:* {b_desc['scientific_boundary']}")

        st.markdown("---")

        # Visual Frame Inspector
        st.subheader("🖼️ Visual Frame Behaviour Inspector")
        st.caption("Inspect individual classroom frames with colour-coded bounding boxes and behaviour classification badges.")

        # Get sorted list of extracted frames present in beh_df
        unique_frame_files = sorted(beh_df["frame_filename"].unique())
        if unique_frame_files:
            selected_frame_filename = st.select_slider(
                label="Select Frame to Inspect:",
                options=unique_frame_files,
                value=unique_frame_files[0],
                help="Slide to inspect student observable behaviours frame-by-frame.",
                key="slider_beh_frame_inspect",
            )

            # Get frame predictions
            curr_frame_preds = beh_df[beh_df["frame_filename"] == selected_frame_filename]
            extracted_idx = int(curr_frame_preds["extracted_frame_index"].iloc[0]) if not curr_frame_preds.empty else 1
            timestamp_val = float(curr_frame_preds["timestamp_seconds"].iloc[0]) if not curr_frame_preds.empty else 0.0

            # Render frame image
            frame_img = annotated_dict.get(selected_frame_filename)
            if frame_img is None:
                # Load from disk and annotate on the fly
                f_path = frames_dir / selected_frame_filename
                if f_path.exists():
                    bgr = cv2.imread(str(f_path))
                    if bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        preds_list = [
                            BehaviourPrediction(
                                video_id=r["video_id"],
                                frame_id=r["frame_id"],
                                extracted_frame_index=r["extracted_frame_index"],
                                timestamp_seconds=r["timestamp_seconds"],
                                frame_filename=r["frame_filename"],
                                track_id=r["track_id"],
                                behaviour_class=r["behaviour_class"],
                                confidence=r["confidence"],
                                x1=r["x1"],
                                y1=r["y1"],
                                x2=r["x2"],
                                y2=r["y2"],
                                visual_evidence=r.get("visual_evidence", ""),
                            )
                            for _, r in curr_frame_preds.iterrows()
                        ]
                        frame_img = classifier.draw_behaviours(rgb, preds_list)

            vis_col1, vis_col2 = st.columns([1.3, 1.0], gap="medium")

            with vis_col1:
                st.markdown(f"**Frame:** `{selected_frame_filename}` | **Index:** `{extracted_idx}` | **Time:** `{timestamp_val:.3f}s`")
                if frame_img is not None:
                    st.image(frame_img, caption=f"Classified Observable Behaviours in {selected_frame_filename}", use_container_width=True)
                else:
                    st.warning("Frame preview unavailable.")

            with vis_col2:
                st.markdown("##### 👥 Tracked People in this Frame")
                if not curr_frame_preds.empty:
                    display_cols = ["track_id", "behaviour_class", "confidence", "visual_evidence"]
                    table_df = curr_frame_preds[display_cols].rename(columns={
                        "track_id": "Track ID",
                        "behaviour_class": "Observed Behaviour",
                        "confidence": "Confidence",
                        "visual_evidence": "Visual Evidence",
                    })
                    st.dataframe(table_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No tracked people in this frame.")

        st.markdown("---")

        # Track-Level Chronological Sequence Viewer
        st.subheader("⏱️ Track-Level Chronological Behaviour Sequence")
        st.info(
            "ℹ️ **Note on Temporal Sequence Display:** This section visualizes simple frame-by-frame observable behaviour records "
            "for an individual Track ID over time. This **does NOT** constitute temporal sequence modeling. "
            "Temporal models (CNN feature extraction, RNN, LSTM, GRU) are strictly reserved for upcoming features."
        )

        available_tracks = sorted(beh_df["track_id"].unique())
        selected_track_id = st.selectbox(
            "Select Student Track ID to inspect over time:",
            options=available_tracks,
            format_func=lambda tid: f"Track ID {tid}",
            key="sb_track_temporal_inspect",
        )

        track_history = beh_df[beh_df["track_id"] == selected_track_id].sort_values("extracted_frame_index")
        if not track_history.empty:
            t_col1, t_col2 = st.columns([1.0, 1.2], gap="medium")

            with t_col1:
                st.markdown(f"##### Summary for Track ID {selected_track_id}")
                st.markdown(f"- **Frames Active:** `{len(track_history)} frames`")
                st.markdown(f"- **First Seen:** `{track_history['timestamp_seconds'].min():.2f}s` (Frame {track_history['extracted_frame_index'].min()})")
                st.markdown(f"- **Last Seen:** `{track_history['timestamp_seconds'].max():.2f}s` (Frame {track_history['extracted_frame_index'].max()})")
                dom_beh = track_history[track_history["behaviour_class"] != CLASS_UNKNOWN]["behaviour_class"].mode()
                st.markdown(f"- **Dominant Observable Category:** `{dom_beh.iloc[0] if not dom_beh.empty else 'Unknown'}`")

            with t_col2:
                st.markdown("##### Chronological Observation Log")
                log_df = track_history[["extracted_frame_index", "timestamp_seconds", "behaviour_class", "confidence"]].rename(columns={
                    "extracted_frame_index": "Frame #",
                    "timestamp_seconds": "Time (s)",
                    "behaviour_class": "Observable Behaviour",
                    "confidence": "Confidence",
                })
                st.dataframe(log_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Behaviours Dataset Preview & CSV Download
        st.subheader("📄 Behaviours Dataset (`behaviours.csv`)")
        st.caption("Complete tabular record of observable behaviour classifications per frame and per track.")

        st.dataframe(beh_df, use_container_width=True, hide_index=True)

        csv_bytes = beh_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Behaviours CSV",
            data=csv_bytes,
            file_name=f"{video_id}_behaviours.csv",
            mime="text/csv",
            help="Download complete observable behaviour recognition dataset.",
            key="btn_download_behaviours_csv",
        )


def render_cnn_feature_extraction_section(saved_path: Path):
    """Render Feature 6: CNN Visual Feature Extraction Section."""
    video_id = derive_video_id(saved_path)
    frames_dir = FRAMES_DIR / video_id
    frame_meta_path = PROCESSED_DIR / video_id / "frame_metadata.csv"
    tracks_csv_path = PROCESSED_DIR / video_id / TRACKS_CSV_FILENAME
    behaviours_csv_path = PROCESSED_DIR / video_id / BEHAVIOURS_CSV_FILENAME
    cnn_npy_path = PROCESSED_DIR / video_id / CNN_FEATURES_NPY_FILENAME
    cnn_meta_path = PROCESSED_DIR / video_id / CNN_METADATA_CSV_FILENAME

    st.header("🧠 Feature 6: CNN Visual Feature Extraction")
    st.caption(
        "Converts each tracked student's visual image crop into a fixed 512-dimensional numerical vector "
        "using a pretrained ResNet18 convolutional backbone with the final classification layer removed."
    )

    st.info(
        "🔬 **Research Scope & Ethical Boundaries:** The CNN operates strictly as a visual feature extractor "
        "(encoding body posture, head orientation, visible objects/desks, and spatial appearance). "
        "It does **not** detect or infer internal mental states such as motivation, boredom, intelligence, "
        "understanding, or cognitive engagement."
    )

    # Verify Feature 2 and Feature 4 outputs exist
    if not frame_meta_path.exists():
        st.warning(
            "⚠️ Preprocessed video frames not found. Please complete **Feature 2: Frame Extraction & Preprocessing** above."
        )
        return

    if not tracks_csv_path.exists():
        st.warning(
            "⚠️ Tracking data (`tracks.csv`) not found. Please complete **Feature 4: Student / Person Tracking** above."
        )
        return

    try:
        frames_df = pd.read_csv(frame_meta_path)
    except Exception as exc:
        st.error(f"❌ Failed to load frame metadata: {exc}")
        return

    try:
        tracks_df = pd.read_csv(tracks_csv_path)
    except Exception as exc:
        st.error(f"❌ Failed to load tracking dataset: {exc}")
        return

    behaviours_df = None
    if behaviours_csv_path.exists():
        try:
            behaviours_df = pd.read_csv(behaviours_csv_path)
        except Exception:
            behaviours_df = None

    extractor = get_cnn_feature_extractor()

    # Controls Layout
    st.subheader("⚙️ Feature Extraction Controls")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1.0, 1.0, 1.0])

    with ctrl_col1:
        st.selectbox(
            label="Pretrained CNN Model",
            options=SUPPORTED_CNN_MODELS,
            index=0,
            help="Pretrained PyTorch ResNet18 backbone. Final fully connected classification layer is replaced with Identity for fixed 512-dim visual representation.",
            key="cnn_model_selection",
        )
    with ctrl_col2:
        st.metric(
            label="Hardware Device",
            value=extractor.device_name,
            help="Automatically detected compute hardware (CUDA if GPU available, otherwise CPU).",
        )
    with ctrl_col3:
        st.metric(
            label="Feature Dimension",
            value=f"{extractor.feature_dim}D",
            help="Fixed size of the output numerical vector produced by the CNN backbone.",
        )
    with ctrl_col4:
        batch_size = st.selectbox(
            label="Batch Size",
            options=[8, 16, 32, 64],
            index=1,
            help="Number of student crops processed simultaneously in each forward pass.",
            key="cnn_batch_size",
        )

    # Frame Range Selection
    total_frames = len(frames_df)
    frame_mode = st.radio(
        label="Frames to process:",
        options=["All extracted frames", "Sample frames (first N)", "Custom frame range"],
        index=0,
        horizontal=True,
        key="cnn_frame_selection_mode",
    )

    first_n = 10
    r_start = 1
    r_end = min(total_frames, 18)

    if frame_mode == "Sample frames (first N)":
        first_n = st.slider(
            "Select first N frames to process:",
            min_value=1,
            max_value=max(1, total_frames),
            value=min(10, max(1, total_frames)),
            key="cnn_sample_n",
        )
        mode_key = "sample"
    elif frame_mode == "Custom frame range":
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            r_start = st.number_input(
                "Start Frame Index:",
                min_value=1,
                max_value=max(1, total_frames),
                value=1,
                key="cnn_range_start",
            )
        with r_c2:
            r_end = st.number_input(
                "End Frame Index:",
                min_value=int(r_start),
                max_value=max(1, total_frames),
                value=min(max(1, total_frames), int(r_start) + 15),
                key="cnn_range_end",
            )
        mode_key = "range"
    else:
        mode_key = "all"

    # Extraction Button
    if st.button(
        "🚀 Extract CNN Visual Features",
        key="btn_run_cnn_extraction",
        type="primary",
        use_container_width=True,
    ):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(prog, msg):
            progress_bar.progress(prog)
            status_text.text(msg)

        with st.spinner("Extracting CNN visual features from tracked person crops..."):
            success, summary, meta_df, feats_arr, err_msg = run_cnn_feature_extraction(
                video_id=video_id,
                frames_df=frames_df,
                tracks_df=tracks_df,
                feature_extractor=extractor,
                behaviours_df=behaviours_df,
                batch_size=int(batch_size),
                frame_selection_mode=mode_key,
                first_n=int(first_n),
                range_start=int(r_start),
                range_end=int(r_end),
                progress_callback=update_progress,
            )

        if success and summary:
            st.success(
                f"✅ **CNN Feature Extraction Complete:** {summary.valid_features_extracted} feature vectors "
                f"extracted in {summary.processing_time_seconds:.2f}s."
            )
            st.session_state[f"{video_id}_cnn_summary"] = summary
        else:
            st.error(f"❌ **Extraction Failed:** {err_msg}")

    # Display Existing / Newly Generated CNN Features
    if cnn_npy_path.exists() and cnn_meta_path.exists():
        try:
            features_array = np.load(str(cnn_npy_path))
            meta_df = pd.read_csv(cnn_meta_path)
        except Exception as exc:
            st.error(f"❌ Error loading CNN feature files from disk: {exc}")
            return

        st.markdown("---")
        st.subheader("📊 CNN Feature Extraction Summary")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Frames Processed", f"{meta_df['extracted_frame_index'].nunique():,}")
        with m2:
            st.metric("Unique Tracks", f"{meta_df['track_id'].nunique():,}")
        with m3:
            st.metric("Valid Feature Vectors", f"{len(features_array):,}")
        with m4:
            st.metric("Feature Dimension", f"{features_array.shape[1] if features_array.ndim == 2 else 512}D")
        with m5:
            st.metric("CNN Model", extractor.model_name.capitalize())
        with m6:
            st.metric("Device Used", extractor.device_name)

        st.markdown("---")

        # Interactive Visual Inspection: Frame -> Track -> Crop -> Feature Preview
        st.subheader("🔍 Visual Feature Inspection")
        st.caption(
            "Select a processed classroom frame and a tracked student to inspect their visual crop and "
            "the corresponding 512-dimensional CNN feature vector."
        )

        avail_frame_indices = sorted(meta_df["extracted_frame_index"].unique())
        if not avail_frame_indices:
            st.info("No processed frame feature vectors available.")
            return

        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            selected_frame_idx = st.selectbox(
                "Select Frame Index for Inspection",
                avail_frame_indices,
                index=0,
                key="cnn_inspect_frame_idx",
            )

        frame_features = meta_df[meta_df["extracted_frame_index"] == selected_frame_idx]
        avail_tracks = sorted(frame_features["track_id"].unique())

        with sel_col2:
            selected_track_id = st.selectbox(
                "Select Track ID",
                avail_tracks,
                index=0,
                key="cnn_inspect_track_id",
            )

        # Retrieve selected crop metadata and vector
        target_rows = frame_features[frame_features["track_id"] == selected_track_id]
        if target_rows.empty:
            st.info("No features found for this track in the selected frame.")
            return
        target_row = target_rows.iloc[0]
        feat_idx = int(target_row["feature_index"])
        feature_vector = features_array[feat_idx]

        # Load original frame and crop
        frame_filename = str(target_row["frame_filename"])
        frame_file_path = frames_dir / frame_filename
        crop_image = None

        if frame_file_path.exists():
            bgr_frame = cv2.imread(str(frame_file_path))
            if bgr_frame is not None:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                bbox = (float(target_row["x1"]), float(target_row["y1"]), float(target_row["x2"]), float(target_row["y2"]))
                from src.behaviour.preprocessing import preprocess_person_crop
                success_crop, crop_image, _, _ = preprocess_person_crop(rgb_frame, bbox)

                # Draw bounding box on full frame preview
                annotated_frame = rgb_frame.copy()
                x1, y1, x2, y2 = int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (46, 204, 113), 3)
                cv2.putText(
                    annotated_frame,
                    f"Track ID {selected_track_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (46, 204, 113),
                    2,
                    cv2.LINE_AA,
                )

        # Two-column inspection layout
        crop_col, feat_col = st.columns([1.0, 1.2], gap="large")

        with crop_col:
            st.markdown(f"##### Track ID {selected_track_id} Crop (Frame #{selected_frame_idx})")
            if crop_image is not None:
                st.image(
                    crop_image,
                    caption=f"Track ID {selected_track_id} — 224x224 RGB Person Crop",
                    use_container_width=True,
                )
            else:
                st.warning("Crop image preview could not be generated.")

            st.markdown(f"- **Timestamp:** `{target_row['timestamp_seconds']:.2f}s`")
            st.markdown(f"- **Bounding Box:** `[{target_row['x1']}, {target_row['y1']}, {target_row['x2']}, {target_row['y2']}]`")
            st.markdown(f"- **Tracking Confidence:** `{target_row['confidence']:.2f}`")
            st.markdown(f"- **Linked Observable Behaviour:** `{target_row['behaviour_class']}`")

        with feat_col:
            st.markdown("##### 512-Dimensional Feature Representation")
            st.markdown(f"- **Model:** `ResNet18 (ImageNet Pretrained)`")
            st.markdown(f"- **Vector Index in Array:** `{feat_idx}`")
            st.markdown(f"- **Output Dimension:** `{len(feature_vector)}`")

            # Numerical statistical properties
            norm_val = float(np.linalg.norm(feature_vector))
            mean_val = float(np.mean(feature_vector))
            std_val = float(np.std(feature_vector))
            min_val = float(np.min(feature_vector))
            max_val = float(np.max(feature_vector))

            st.markdown("###### Feature Vector Statistics")
            stat_c1, stat_c2, stat_c3, stat_c4, stat_c5 = st.columns(5)
            stat_c1.metric("L2 Norm", f"{norm_val:.2f}")
            stat_c2.metric("Mean", f"{mean_val:.4f}")
            stat_c3.metric("Std", f"{std_val:.4f}")
            stat_c4.metric("Min", f"{min_val:.4f}")
            stat_c5.metric("Max", f"{max_val:.4f}")

            st.markdown("###### First 10 Numerical Values (Preview)")
            preview_values = [round(float(v), 5) for v in feature_vector[:10]]
            st.code(f"{preview_values}\n... [{len(feature_vector) - 10} additional dimensions truncated]", language="python")

        st.markdown("---")

        # 2D PCA Feature Space Visualization
        st.subheader("📊 2D PCA Feature Space Distribution")
        st.caption(
            "Visualizes similarity in extracted visual feature space only. "
            "Does NOT prove or infer mental engagement, cognitive focus, motivation, or boredom."
        )

        pca_pts, var_exp = compute_pca_2d(features_array)
        if pca_pts is not None and len(pca_pts) >= 2:
            pca_color_option = st.radio(
                "Color PCA Points By:",
                ["Track ID", "Observable Behaviour"],
                index=0,
                horizontal=True,
                key="cnn_pca_color_option",
            )
            color_key = "behaviour_class" if pca_color_option == "Observable Behaviour" else "track_id"
            fig = create_pca_scatter_figure(
                projected=pca_pts,
                metadata_df=meta_df,
                color_by=color_key,
                explained_variance=var_exp,
            )
            st.pyplot(fig)
        else:
            st.info("Insufficient feature samples (minimum 2 required) to compute 2D PCA projection.")

        st.markdown("---")

        # Features Dataset Preview & Downloads
        st.subheader("📄 CNN Features Metadata (`cnn_features_metadata.csv`)")
        st.caption(
            "Complete tabular metadata mapping each row index of `cnn_features.npy` to its corresponding "
            "frame ID, timestamp, Track ID, bounding box, and linked observable behaviour."
        )

        st.dataframe(meta_df, use_container_width=True, hide_index=True)

        down_col1, down_col2 = st.columns(2)
        with down_col1:
            csv_bytes = meta_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CNN Metadata CSV",
                data=csv_bytes,
                file_name=f"{video_id}_{CNN_METADATA_CSV_FILENAME}",
                mime="text/csv",
                help="Download tabular mapping linking feature vector indices to frame, track, and behaviour metadata.",
                key="btn_download_cnn_metadata_csv",
            )

        with down_col2:
            import io
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, features_array)
            npy_bytes = npy_buffer.getvalue()
            st.download_button(
                label="📥 Download CNN Features Array (.npy)",
                data=npy_bytes,
                file_name=f"{video_id}_{CNN_FEATURES_NPY_FILENAME}",
                mime="application/octet-stream",
                help="Download raw float32 NumPy binary feature array of shape (N, 512).",
                key="btn_download_cnn_features_npy",
            )


def render_temporal_sequence_section(saved_path: Path):
    """Render Feature 7 Temporal Sequence Creation section."""
    st.subheader("⏱️ Feature 7: Temporal Sequence Creation")
    st.caption(
        "Organizes extracted CNN visual feature embeddings chronologically into fixed-length sliding-window "
        "sequences for each tracked student. Prepares PyTorch-ready 3D tensors (N, L, D) for downstream sequence modeling."
    )

    with st.expander("ℹ️ About Temporal Sequence Creation & Academic Scope", expanded=False):
        st.markdown(
            """
            - **Track-Wise Isolation:** Features are partitioned strictly by Track ID so observations from different students are never mixed.
            - **Chronological Ordering:** Observations within each track are strictly sorted by extracted frame index and video timestamp.
            - **Sliding-Window Chunking:** Continuous track segments are partitioned into overlapping fixed-length windows of length $L$ and stride $S$.
            - **Gap Splitting:** If the tracking gap between consecutive observations exceeds tolerance ($G$ frames), the segment is safely split to prevent unobserved jumps.
            - **Downstream Compatibility:** Decoupled into a 3D NumPy array `temporal_sequences.npy` of shape `(N, L, D)` and a spatial-temporal metadata ledger `temporal_sequences_metadata.csv`, with direct PyTorch `Dataset` and `DataLoader` compatibility.
            - **Strict Research Boundaries:** Sequence creation organizes observable visual features across time. It does **not** perform RNN/LSTM training or inference (Feature 8), nor does it claim or infer internal mental states.
            """
        )

    video_id = derive_video_id(saved_path)
    processed_dir = PROCESSED_DIR / video_id

    cnn_npy_path = processed_dir / CNN_FEATURES_NPY_FILENAME
    cnn_meta_path = processed_dir / CNN_METADATA_CSV_FILENAME
    seq_npy_path = processed_dir / TEMPORAL_SEQUENCES_NPY_FILENAME
    seq_meta_path = processed_dir / TEMPORAL_SEQUENCES_METADATA_FILENAME

    # Verify Feature 6 prerequisites
    if not cnn_npy_path.exists() or not cnn_meta_path.exists():
        st.warning(
            "⚠️ **Feature 6 (CNN Visual Feature Extraction) is required before temporal sequences can be created.** "
            "Please run Feature 6 above to generate visual feature embeddings."
        )
        return

    # Configuration controls
    st.markdown("##### ⚙️ Sequence Generation Parameters")
    param_col1, param_col2, param_col3 = st.columns(3)

    with param_col1:
        seq_length = st.slider(
            "Sequence Length (Frames / Time Steps $L$)",
            min_value=3,
            max_value=30,
            value=DEFAULT_SEQUENCE_LENGTH,
            step=1,
            help="Number of consecutive observations in each temporal sequence window.",
            key="temporal_seq_length_slider",
        )

    with param_col2:
        seq_stride = st.slider(
            "Window Stride ($S$)",
            min_value=1,
            max_value=10,
            value=DEFAULT_SEQUENCE_STRIDE,
            step=1,
            help="Step size between consecutive sliding windows. Stride < Length produces overlapping sequences.",
            key="temporal_stride_slider",
        )

    with param_col3:
        max_gap = st.slider(
            "Max Frame Gap Tolerance ($G$)",
            min_value=1,
            max_value=10,
            value=DEFAULT_MAX_FRAME_GAP,
            step=1,
            help="Maximum allowable missing frame count before a track is split into separate continuous segments.",
            key="temporal_max_gap_slider",
        )

    # Action button
    if st.button("🚀 Create Temporal Sequences", type="primary", use_container_width=True, key="btn_run_temporal_sequences"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(prog: float, text: str):
            progress_bar.progress(prog)
            status_text.text(text)

        with st.spinner("Generating temporal sliding-window sequences across student tracks..."):
            success, summary, metadata_df, sequences_array, err_msg = run_temporal_sequence_creation(
                video_id=video_id,
                cnn_features_path=cnn_npy_path,
                cnn_metadata_path=cnn_meta_path,
                sequence_length=int(seq_length),
                stride=int(seq_stride),
                max_frame_gap=int(max_gap),
                progress_callback=update_progress,
            )

        if success and summary:
            st.success(
                f"✅ **Temporal Sequence Creation Complete:** Created {summary.total_sequences_created} sequences "
                f"with tensor shape {summary.tensor_shape} across {summary.valid_tracks_processed} valid student tracks "
                f"in {summary.processing_time_seconds:.2f}s."
            )
            st.session_state[f"{video_id}_temporal_summary"] = summary
        else:
            st.error(f"❌ **Sequence Creation Failed:** {err_msg}")

    # Display Existing / Generated Temporal Sequences
    if seq_npy_path.exists() and seq_meta_path.exists():
        try:
            seq_array = np.load(str(seq_npy_path))
            seq_meta_df = pd.read_csv(seq_meta_path)
        except Exception as exc:
            st.error(f"❌ Error loading temporal sequence files from disk: {exc}")
            return

        st.markdown("---")
        st.subheader("📊 Temporal Sequence Summary")

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Sequences Created", f"{len(seq_array):,}")
        with s2:
            st.metric("Tracks Covered", f"{seq_meta_df['track_id'].nunique():,}")
        with s3:
            st.metric("PyTorch Tensor Shape", f"{seq_array.shape}")
        with s4:
            avg_dur = float(seq_meta_df["duration_seconds"].mean()) if "duration_seconds" in seq_meta_df else 0.0
            st.metric("Avg Duration", f"{avg_dur:.2f}s")

        # Config specs
        st.markdown("##### Configuration & Dataset Properties")
        spec_c1, spec_c2 = st.columns(2)
        with spec_c1:
            st.markdown(
                f"""
                - **Sequence Length ($L$):** `{seq_array.shape[1] if seq_array.ndim == 3 else seq_length} frames`
                - **Window Stride ($S$):** `{seq_stride}`
                - **Feature Dimension ($D$):** `{seq_array.shape[2] if seq_array.ndim == 3 else 512}`
                """
            )
        with spec_c2:
            st.markdown(
                f"""
                - **Sequences Array:** `{seq_npy_path.name}` (`{seq_array.nbytes / (1024 * 1024):.2f} MB`)
                - **Sequences Metadata:** `{seq_meta_path.name}` (`{len(seq_meta_df)} rows`)
                - **PyTorch DataType:** `torch.float32`
                """
            )

        st.markdown("---")

        # Interactive Sequence Inspector
        st.subheader("🔍 Interactive Sequence Inspector")
        st.caption("Select a generated temporal sequence to inspect its temporal window, frame indices, and feature norm progression.")

        avail_seq_ids = sorted(seq_meta_df["sequence_id"].tolist())
        if avail_seq_ids:
            sel_seq_id = st.selectbox(
                "Select Sequence ID to Inspect",
                avail_seq_ids,
                index=0,
                format_func=lambda sid: f"Sequence #{sid} (Track ID {seq_meta_df.loc[seq_meta_df['sequence_id'] == sid, 'track_id'].iloc[0]} | Dominant: {seq_meta_df.loc[seq_meta_df['sequence_id'] == sid, 'dominant_behaviour'].iloc[0]})",
                key="temporal_inspect_seq_id",
            )

            seq_row = seq_meta_df[seq_meta_df["sequence_id"] == sel_seq_id].iloc[0]
            seq_feature_slice = seq_array[sel_seq_id]

            info_col, plot_col = st.columns([1.0, 1.3], gap="medium")
            with info_col:
                st.markdown("##### Sequence Properties")
                st.markdown(f"- **Track ID:** `{int(seq_row['track_id'])}`")
                st.markdown(f"- **Extracted Frame Range:** `Frame {int(seq_row['start_extracted_frame_index'])} → Frame {int(seq_row['end_extracted_frame_index'])}`")
                st.markdown(f"- **Video Time Range:** `{float(seq_row['start_timestamp_seconds']):.2f}s → {float(seq_row['end_timestamp_seconds']):.2f}s` (`{float(seq_row['duration_seconds']):.2f}s`)")
                st.markdown(f"- **Dominant Behaviour:** `{seq_row['dominant_behaviour']}`")
                st.markdown(f"- **Mean Behaviour Confidence:** `{float(seq_row['mean_behaviour_confidence']):.2%}`")
                st.markdown(f"- **Transition Chain:** `{seq_row['behaviour_sequence']}`")

                with st.expander("Frame-by-Frame Breakdown", expanded=False):
                    try:
                        import json
                        f_indices = json.loads(seq_row["frame_indices"]) if isinstance(seq_row["frame_indices"], str) else seq_row["frame_indices"]
                        f_timestamps = json.loads(seq_row["frame_timestamps"]) if isinstance(seq_row["frame_timestamps"], str) else seq_row["frame_timestamps"]
                        breakdown_df = pd.DataFrame({
                            "Step": [f"t{i+1}" for i in range(len(f_indices))],
                            "Extracted Frame": f_indices,
                            "Timestamp (s)": f_timestamps,
                        })
                        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.caption(f"Details: {e}")

            with plot_col:
                fig = create_sequence_timeline_figure(seq_feature_slice, seq_row.to_dict())
                st.pyplot(fig)

        st.markdown("---")

        # Track Coverage Timeline Chart
        st.subheader("📈 Track Temporal Window Coverage")
        st.caption("Illustrates the time windows covered by temporal sequences across each tracked student.")
        cov_fig = create_track_coverage_figure(seq_meta_df)
        if cov_fig is not None:
            st.pyplot(cov_fig)

        st.markdown("---")

        # PyTorch Dataset Compatibility Demonstration
        st.subheader("🔌 PyTorch DataLoader & Model Integration (Ready for Feature 8)")
        st.caption("How downstream models in Feature 8 can directly load these temporal sequences using standard PyTorch utilities:")
        st.code(
            f"""from torch.utils.data import DataLoader
from src.temporal import ClassroomSequenceDataset
import numpy as np
import pandas as pd

# 1. Load generated sequence array and metadata
sequences = np.load("{seq_npy_path}")  # Shape: {seq_array.shape}
metadata_df = pd.read_csv("{seq_meta_path}")

# 2. Instantiate PyTorch Dataset adapter
dataset = ClassroomSequenceDataset(sequences=sequences, metadata_df=metadata_df)

# 3. Create standard PyTorch DataLoader with batching & shuffling
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. In Feature 8: Iterate batches (batch_shape: [8, {seq_array.shape[1]}, {seq_array.shape[2]}])
for batch_tensors, batch_metadata in dataloader:
    # Model forward pass: outputs = model(batch_tensors)
    pass
""",
            language="python",
        )

        st.markdown("---")

        # Dataset Preview & Downloads
        st.subheader("📄 Temporal Sequences Metadata (`temporal_sequences_metadata.csv`)")
        st.caption(
            "Complete tabular metadata mapping each sequence index to its Track ID, frame window, "
            "timestamp range, dominant observable behaviour, and frame indices."
        )

        st.dataframe(seq_meta_df, use_container_width=True, hide_index=True)

        down_col1, down_col2 = st.columns(2)
        with down_col1:
            csv_bytes = seq_meta_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Sequences Metadata CSV",
                data=csv_bytes,
                file_name=f"{video_id}_{TEMPORAL_SEQUENCES_METADATA_FILENAME}",
                mime="text/csv",
                help="Download tabular mapping linking temporal sequence indices to tracks, frame windows, and behaviours.",
                key="btn_download_temporal_metadata_csv",
            )

        with down_col2:
            import io
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, seq_array)
            npy_bytes = npy_buffer.getvalue()
            st.download_button(
                label="📥 Download Sequences Array (.npy)",
                data=npy_bytes,
                file_name=f"{video_id}_{TEMPORAL_SEQUENCES_NPY_FILENAME}",
                mime="application/octet-stream",
                help="Download raw float32 NumPy 3D tensor of shape (N, L, D).",
                key="btn_download_temporal_sequences_npy",
            )


def main():
    """Main application loop."""
    render_sidebar()
    render_header()

    st.subheader("📁 Upload Classroom Video")

    # Supported extension list formatted for st.file_uploader
    supported_types = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]

    uploaded_file = st.file_uploader(
        label="Select a classroom video file to upload and validate",
        type=supported_types,
        help="Supported formats: MP4, AVI, MOV, MKV. Maximum recommended size: 1GB.",
        key="classroom_video_uploader",
    )

    if uploaded_file is None:
        st.info(
            "👉 Please select a classroom video file above to begin validation and preview."
        )
        return

    # Check file extension explicitly
    is_valid_ext, ext_err = validate_file_extension(uploaded_file.name)
    if not is_valid_ext:
        st.error(f"❌ {ext_err}")
        return

    # Process and save video
    with st.spinner("Validating video file format, decodability, and extracting stream metadata..."):
        success, saved_path, metadata, message = save_uploaded_video(
            uploaded_file, target_dir=VIDEOS_DIR
        )

    if not success:
        st.error(f"❌ **Validation Failed:** {message}")
        st.caption("Ensure the uploaded file is a complete, uncorrupted video with valid video streams.")
        return

    # Video successfully verified
    st.success(f"✅ **Status:** {message}")

    st.markdown("---")

    # Layout: Preview & Metadata
    preview_col, meta_col = st.columns([1.1, 1.0], gap="large")

    with preview_col:
        st.subheader("🎬 Video Preview")
        if saved_path and saved_path.exists():
            st.video(str(saved_path))
        else:
            st.warning("Video file preview is temporarily unavailable on disk.")

    with meta_col:
        if metadata:
            render_metadata_section(metadata)

    st.markdown("---")

    # Feature 2: Frame Extraction & Preprocessing Section
    if saved_path and saved_path.exists() and metadata:
        render_frame_extraction_section(saved_path, metadata)

    st.markdown("---")

    # Feature 3: Student / Person Detection Section
    if saved_path and saved_path.exists():
        render_detection_section(saved_path)

    st.markdown("---")

    # Feature 4: Student / Person Tracking Section
    if saved_path and saved_path.exists():
        render_tracking_section(saved_path)

    st.markdown("---")

    # Feature 5: Observable Behaviour Recognition Section
    if saved_path and saved_path.exists():
        render_behaviour_recognition_section(saved_path)

    st.markdown("---")

    # Feature 6: CNN Visual Feature Extraction Section
    if saved_path and saved_path.exists():
        render_cnn_feature_extraction_section(saved_path)

    st.markdown("---")

    # Feature 7: Temporal Sequence Creation Section
    if saved_path and saved_path.exists():
        render_temporal_sequence_section(saved_path)


if __name__ == "__main__":
    main()


