"""EduPulse AI - Temporal Learning-Engagement Profiling from Classroom Videos.

Feature 1: Project Setup + Classroom Video Input & Metadata Profiling.
Feature 2: Frame Extraction & Preprocessing with Temporal Metadata Preservation.
"""

from pathlib import Path
import sys

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import math

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_SAMPLING_INTERVAL,
    EXCLUDED_INTERNAL_STATES,
    FRAMES_DIR,
    PROCESSED_DIR,
    PROJECT_TITLE,
    SUPPORTED_EXTENSIONS,
    TARGET_OBSERVABLE_BEHAVIOURS,
    VIDEOS_DIR,
    ensure_directories,
)
from src.preprocessing.frame_preprocessor import FramePreprocessor
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
    page_title="EduPulse AI | Classroom Video Input & Frame Extraction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure data directories exist
ensure_directories()


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
        st.success("🚀 **Feature 2: Frame Extraction & Preprocessing**")
        st.caption("Next stages (Detection, Tracking, Behaviour CNN/RNN) unlock in future milestones.")


def render_header():
    """Render main application header."""
    st.title(PROJECT_TITLE)
    st.markdown(
        "Upload a classroom recording (`.mp4`, `.avi`, `.mov`, `.mkv`), inspect stream properties, "
        "and extract chronological preprocessed frames for temporal engagement profiling."
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


if __name__ == "__main__":
    main()
