# Temporal Learning-Engagement Profiling from Classroom Videos

An educational research and computer vision system to analyze classroom video recordings and profile **observable learning-related student behaviours over time**.

---

## 1. Project Overview & Research Purpose

The primary objective of this project is to analyze classroom video streams and identify **observable physical and behavioral indicators** of student classroom engagement across temporal sequences.

### Target Observable Behaviours
1. **Looking toward the instructional activity** (board, slides, educator)
2. **Reading / writing** (taking notes, following course material)
3. **Interacting with peers** (collaborative coursework, discussions)
4. **Looking away** (distraction outside learning focus)
5. **Mobile-device activity** (unauthorized or off-task phone/screen usage)
6. **Head-down behaviour** (resting head on desk, sleeping posture)

### Ethical Boundaries & Scope Exclusions
In accordance with ethical pedagogical standards and computer vision validity, this system **strictly rejects subjective claims of internal mental states**. 

The system **DOES NOT** detect or predict:
* Emotions (e.g., happiness, sadness, anger)
* Subjective states (e.g., boredom, motivation, interest)
* Student intelligence or comprehension
* Internal cognitive load or mental engagement/disengagement
* Student personal identity or facial recognition

The system focuses exclusively on verifiable, observable physical behaviours recorded in educational spaces.

---

## 2. Technology Stack (Feature 1)

* **Programming Language:** Python 3.12+ (supports Python 3.11+)
* **Web Application Framework:** Streamlit
* **Computer Vision & Video Processing:** OpenCV (`opencv-python`)
* **Numerical Computing:** NumPy
* **Data Structures & Processing:** Pandas
* **Test Suite:** PyTest

*(Future deep learning libraries such as PyTorch, YOLO, and Scikit-Learn will be introduced in subsequent feature milestones).*

---

## 3. Project Structure

```text
EduPulse_AI/
│
├── app.py                          # Streamlit main application entry point
│
├── data/                           # Data storage (git-ignored for student privacy)
│   ├── videos/                     # Uploaded raw classroom videos
│   ├── frames/                     # Extracted and sampled video frames (<video_id>/)
│   └── processed/                  # Processed temporal frame metadata (<video_id>/)
│
├── models/                         # Trained model weights & checkpoints (future)
├── notebooks/                      # Research & exploratory notebooks
│
├── src/                            # Modular source code
│   ├── __init__.py
│   ├── config.py                   # Central paths, sampling defaults, & constraints
│   ├── video/                      # Video ingestion & frame extraction
│   │   ├── __init__.py
│   │   ├── video_utils.py          # Video validation, metadata extraction, sanitization
│   │   └── frame_extractor.py      # Chronological extraction, sampling, & metadata engine
│   ├── preprocessing/              # Image validation, color handling, & resizing
│   │   ├── __init__.py
│   │   └── frame_preprocessor.py   # FramePreprocessor utility class
│   ├── detection/                  # Student detection (future)
│   ├── tracking/                   # Student tracking (future)
│   ├── behaviour/                  # Observable behaviour classification (future)
│   ├── cnn/                        # Spatial visual feature extraction (future)
│   └── temporal/                   # Sequence modeling (RNN/LSTM/GRU) (future)
│
├── results/                        # Evaluation logs and ablation outputs (future)
├── tests/                          # Automated unit and integration tests
│   ├── __init__.py
│   ├── test_app.py                 # Streamlit UI integration tests (Feature 1 & 2)
│   ├── test_video_utils.py         # Video validation unit tests
│   └── test_frame_extractor.py     # Frame extraction, sampling, & preprocessor tests
│
├── requirements.txt                # Core dependencies
├── .gitignore                      # Git ignore rules for video data & environment
└── README.md                       # Project documentation
```

---

## 4. Setup and Installation

### Prerequisites
* Python 3.11 or Python 3.12 installed on your system.
* Git installed.

### Step 1 — Clone the Repository
```bash
git clone https://github.com/hemalatha630/EduPulse_AI.git
cd EduPulse_AI
```

### Step 2 — Create a Virtual Environment
On Windows:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Running the Application

Launch the Streamlit web application with:

```bash
streamlit run app.py
```

Open your web browser and navigate to the local address displayed in your terminal (typically `http://localhost:8501`).

---

## 6. Video Ingestion & Metadata Extraction (Feature 1)

### How to Upload a Classroom Video
1. Open the web interface at `http://localhost:8501`.
2. Locate the **Upload Classroom Video** section.
3. Click **Browse files** or drag and drop a classroom video file.
4. Supported video file containers:
   * `.mp4`
   * `.avi`
   * `.mov`
   * `.mkv`

### Automatic Validation Pipeline
Upon file selection, the backend executes rigorous multi-stage validation:
1. **Extension Verification:** Rejects unsupported file types before ingestion.
2. **Filename Sanitization:** Neutralizes path-traversal patterns (`../../`) and reserved operating system characters.
3. **Storage Persistence:** Saves the file safely into the local `data/videos/` directory (excluded from Git).
4. **OpenCV Decodability Check:** Opens the video file with `cv2.VideoCapture` and decodes the initial frame. Rejects empty (0-byte) or corrupted/spoofed files without displaying technical tracebacks.

### Displayed Video Metadata
Once verified, the interface displays:
* **Filename:** Safe local file name
* **Container Format:** Video format (MP4, AVI, MOV, MKV)
* **File Size:** Human-readable size (KB, MB, GB)
* **Duration:** Calculated exact duration formatted as `MM:SS` (or `HH:MM:SS`)
* **Resolution:** Width × Height (in pixels)
* **Frame Rate (FPS):** Extracted frames per second
* **Total Frames:** Total frame count decoded from the container

---

## 7. Frame Extraction & Preprocessing (Feature 2)

### What is a Video Frame?
A video recording is not a single continuous entity; it consists of a rapid chronological sequence of static photographic images called **frames**, displayed in quick succession (typically 24 to 30 frames every second). Each frame captures the complete physical spatial layout of the classroom at that exact fraction of a second.

### Why Are Frames Extracted?
Computer vision models (such as YOLO for spatial detection and Convolutional Neural Networks for visual feature extraction) do not process raw compressed video container formats (`.mp4`, `.avi`) directly. Instead, they require discrete, decoded pixel arrays (image matrices). Extracting frames decouples video decoding from downstream deep learning pipelines.

### What Does Frame Sampling Mean?
Extracting every frame from a 60-minute classroom video at 30 FPS would yield 108,000 high-resolution images, creating immense disk storage demands and redundant computational overhead, as human classroom postures rarely change radically within 33 milliseconds. 

**Sampling** selects frames at a structured stride (e.g., every 5th frame), converting raw high-FPS video into an **effective sampling rate**:

$$\text{Effective FPS} = \frac{\text{Original Video FPS}}{\text{Sampling Interval}} \approx \frac{30}{5} = 6\text{ FPS}$$

This preserves granular temporal dynamics and behavioral shifts while reducing data volume and computation by 80%.

### Where Are Frames Stored?
Frames are saved in a structured, video-specific directory:
```text
data/
└── frames/
    └── <video_id>/
        ├── frame_000001.jpg
        ├── frame_000002.jpg
        ├── frame_000003.jpg
        └── ...
```
All paths are kept strictly relative and platform-independent. Video directories are automatically ignored by Git to ensure student privacy.

### What Metadata Is Stored?
Each extraction produces a chronological index stored in:
```text
data/processed/<video_id>/frame_metadata.csv
```
The CSV provides complete temporal traceability with the following fields:
* `video_id`: Unique identifier derived safely from the video filename
* `frame_index`: 0-indexed position in the original source video stream
* `extracted_frame_index`: 1-indexed sequential frame number in the extracted subset
* `timestamp_seconds`: Exact elapsed time (in seconds, rounded to 3 decimal places)
* `frame_filename`: Local image file name (`frame_000001.jpg`)
* `frame_path`: Absolute local file system path for CV loaders
* `original_fps`: Source video frames per second
* `sampling_interval`: Frame stride used during extraction
* `width`: Frame pixel width after preprocessing
* `height`: Frame pixel height after preprocessing

### What Preprocessing Is Currently Performed?
1. **Frame Validation:** Validates that each decoded frame is non-empty, three-dimensional (H, W, C), and properly formatted as a valid image array before saving.
2. **Standardized Image Encoding:** Persists frames as standardized JPEG files (`.jpg`) with configurable compression quality (default 95).
3. **Color Space Integrity:** OpenCV reads frames in BGR format by default. The system converts BGR to RGB when displaying frames in Streamlit and preserves native BGR for disk storage, preventing color distortion or channel swapping.
4. **Configurable Resizing:** Provides optional targeted resolution adjustment (e.g., preserving original resolution by default, with presets for 720p HD or 360p Fast) using area interpolation for anti-aliasing.
5. **Caching & Re-extraction Avoidance:** Automatically checks for existing valid extractions matching the selected sampling parameters to avoid duplicate processing overhead.

---

## 8. Automated Testing

Run the automated PyTest suite to verify video ingestion, decodability, frame preprocessing, sampling calculations, temporal ordering, and Streamlit UI workflows:

```bash
pytest tests/ -v
```

The test suite covers:
* `test_video_utils.py`: Filename sanitization, extension validation, OpenCV decodability, empty/corrupt file rejection, metadata extraction.
* `test_frame_extractor.py`: Image validation, color conversion, resizing, chronological timestamp ordering, sampling ratios (1:1, 1:5, 1:10), CSV schema verification, cache handling.
* `test_app.py`: Streamlit end-to-end UI integration tests covering initial render, file upload, metric cards, extraction button triggers, extraction summaries, previews, and corrupted upload handling.

---

## 9. Current Limitations (Feature 1 & Feature 2 Scope)

Features 1 and 2 focus exclusively on **Classroom Video Ingestion, Decodability Validation, Chronological Frame Extraction, Preprocessing, and Temporal Metadata Tracking**.

Current limitations:
* Student bounding box detection (YOLO) is not yet implemented.
* Student tracking across temporal sequences is not yet implemented.
* Observable classroom behaviour classification is not yet implemented.
* Deep learning models (CNN / RNN / LSTM / GRU) are not yet active.
* Cloud / cluster distributed processing is not yet enabled.

---

## 10. Future Research Pipeline Roadmap

The subsequent development phases will follow this structured academic pipeline:

```text
Classroom Video Input (Feature 1 — Completed)
   ↓
Frame Extraction & Preprocessing (Feature 2 — Completed)
   ↓
Student Detection (YOLO / Spatial Bounding Boxes) (Feature 3 — Upcoming)
   ↓
Student Tracking (DeepSORT / ByteTrack Multi-Object Tracking)
   ↓
Observable Behaviour Recognition (Spatial Action Analysis)
   ↓
CNN Visual Feature Extraction (Spatial Representations)
   ↓
Temporal Sequence Creation (Sliding Window Time Sequences)
   ↓
Sequence Modeling (RNN / LSTM / GRU)
   ↓
Observable Behaviour Trajectory Profiling
   ↓
Teaching Activity Correlation Analysis
   ↓
Baseline Model Comparison
   ↓
Ablation Studies & Temporal Error Analysis
   ↓
Final Interactive Analytics Dashboard
```
