"""EduPulse AI - Temporal Learning-Engagement Profiling from Classroom Videos.

Feature 1: Project Setup + Classroom Video Input & Metadata Profiling.
"""

from pathlib import Path
import sys

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.config import (
    EXCLUDED_INTERNAL_STATES,
    PROJECT_TITLE,
    SUPPORTED_EXTENSIONS,
    TARGET_OBSERVABLE_BEHAVIOURS,
    VIDEOS_DIR,
    ensure_directories,
)
from src.video.video_utils import (
    VideoMetadata,
    extract_video_metadata,
    save_uploaded_video,
    validate_file_extension,
)

# Page configuration
st.set_page_config(
    page_title="EduPulse AI | Classroom Video Input",
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
        st.success("**Feature 1: Video Ingestion & Metadata**")
        st.caption("Next stages (Detection, Tracking, Behaviour CNN/RNN) unlock in future milestones.")


def render_header():
    """Render main application header."""
    st.title(PROJECT_TITLE)
    st.markdown(
        "Upload a classroom recording (`.mp4`, `.avi`, `.mov`, `.mkv`) to validate "
        "compatibility, decodability, and inspect video stream properties."
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


if __name__ == "__main__":
    main()
