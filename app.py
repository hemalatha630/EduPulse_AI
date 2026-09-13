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

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL,
    DEFAULT_TRACKER,
    DEFAULT_TRACKING_CONF_THRESHOLD,
    DEFAULT_YOLO_MODEL,
    EXCLUDED_INTERNAL_STATES,
    FRAMES_DIR,
    PROCESSED_DIR,
    PROJECT_TITLE,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_TRACKERS,
    TARGET_OBSERVABLE_BEHAVIOURS,
    TRACKS_CSV_FILENAME,
    VIDEOS_DIR,
    ensure_directories,
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
    page_title="EduPulse AI | Classroom Video Input, Frames, Detection & Tracking",
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
        st.success("🚀 **Feature 4: Student / Person Tracking**")
        st.caption("Next stages (Behaviour Classification, CNN/RNN Modeling) unlock in future milestones.")


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


if __name__ == "__main__":
    main()

