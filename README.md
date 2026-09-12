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

## 2. Technology Stack (Features 1, 2 & 3)

* **Programming Language:** Python 3.12+ (supports Python 3.11+)
* **Web Application Framework:** Streamlit
* **Computer Vision & Video Processing:** OpenCV (`opencv-python`)
* **Object Detection & Deep Learning:** Ultralytics YOLO (`ultralytics`), PyTorch (`torch`, `torchvision`)
* **Image Processing:** Pillow (`Pillow`)
* **Numerical Computing:** NumPy
* **Data Structures & Processing:** Pandas
* **Test Suite:** PyTest

*(Future deep learning modules such as multi-object tracking, CNN behaviour classification, and RNN/LSTM/GRU temporal modeling will be introduced in subsequent feature milestones).*

---

## 3. Project Structure

```text
EduPulse_AI/
│
├── app.py                          # Streamlit main application entry point (Features 1-3)
│
├── data/                           # Data storage (git-ignored for student privacy)
│   ├── videos/                     # Uploaded raw classroom videos
│   ├── frames/                     # Extracted and sampled video frames (<video_id>/)
│   └── processed/                  # Processed metadata and detection outputs (<video_id>/)
│       └── <video_id>/
│           ├── frame_metadata.csv  # Feature 2 chronological frame index
│           └── detections.csv      # Feature 3 per-frame bounding box coordinates
│
├── models/                         # Model weights directory
│   └── yolov8n.pt                  # Pretrained YOLOv8n detector (~6.2 MB)
│
├── notebooks/                      # Research & exploratory notebooks
│
├── src/                            # Modular source code
│   ├── __init__.py
│   ├── config.py                   # Central paths, sampling defaults, & detection configs
│   ├── video/                      # Video ingestion & frame extraction
│   │   ├── __init__.py
│   │   ├── video_utils.py          # Video validation, metadata extraction, sanitization
│   │   └── frame_extractor.py      # Chronological extraction, sampling, & metadata engine
│   ├── preprocessing/              # Image validation, color handling, & resizing
│   │   ├── __init__.py
│   │   └── frame_preprocessor.py   # FramePreprocessor utility class
│   ├── detection/                  # Student / Person Detection (Feature 3)
│   │   ├── __init__.py
│   │   └── detector.py             # YOLOPersonDetector & detection pipeline
│   ├── tracking/                   # Student tracking (Feature 4 — upcoming)
│   ├── behaviour/                  # Observable behaviour classification (future)
│   ├── cnn/                        # Spatial visual feature extraction (future)
│   └── temporal/                   # Sequence modeling (RNN/LSTM/GRU) (future)
│
├── results/                        # Evaluation logs and ablation outputs (future)
├── tests/                          # Automated unit and integration tests
│   ├── __init__.py
│   ├── test_app.py                 # Streamlit UI integration tests (Features 1, 2 & 3)
│   ├── test_video_utils.py         # Video validation unit tests
│   ├── test_frame_extractor.py     # Frame extraction, sampling, & preprocessor tests
│   └── test_detector.py            # YOLO person detection & annotation unit tests
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

## 8. Feature 3 — Student / Person Detection

Feature 3 extends the educational computer vision pipeline by localizing visible people (students and educators) in extracted classroom video frames using a lightweight pretrained YOLO object detector.

### Model Choice & Academic Rationale
* **Model:** YOLOv8n (`yolov8n.pt`, Ultralytics), pretrained on the COCO benchmark.
* **Size & Footprint:** ~6.2 MB disk footprint, 3.2M parameters.
* **Inference Speed:** Real-time on modern GPUs and low latency on university lab commodity CPUs (~50–80 ms per frame).
* **Rationale:** The nano architecture provides an optimal tradeoff between detection accuracy for human postures in lecture halls and computational efficiency, avoiding heavy workstation GPU requirements during classroom analysis.

### Detection vs. Tracking: Fundamental Boundary
* **Feature 3 (Detection):** Performs isolated, frame-by-frame spatial localization. It identifies *where* people are located at each instant $t$, generating bounding box coordinates $(x_1, y_1, x_2, y_2)$ and confidence scores.
* **Feature 4 (Tracking — Future):** Connects detections across time steps $t, t+1, \dots, t+K$ to establish persistent student identities and trajectories using multi-object tracking (ByteTrack / DeepSORT).
* **Isolation Rule:** Feature 3 strictly **DOES NOT** assign tracking IDs or attempt trajectory linking. Each frame's detections are independently indexed (`detection_index = 0, 1, 2, ...`).

### What Classes Are Detected?
* The detector filters exclusively for **COCO Class 0 (`person`)**.
* Detections are labeled generically as **"Detected People"** (students and instructors in the camera field of view).
* All other 79 COCO classes (e.g., chairs, desks, backpacks, bottles, laptops) are filtered out to prevent clutter and false associations.
* **No Face Recognition:** Faces and personal identities are neither segmented nor matched against biometric databases.

### Confidence Threshold Tradeoff
The Streamlit interface provides an interactive confidence threshold slider ($0.10 \le \tau \le 1.00$, default $\tau = 0.25$):
* **Lower Thresholds ($\tau \approx 0.15 - 0.25$):** Prioritize *recall* — essential in university lecture halls where students in distant rows are partially occluded by desk rows or computer screens.
* **Higher Thresholds ($\tau \ge 0.40$):** Prioritize *precision* — useful in close-up seminar rooms to eliminate any false-positive detections on classroom background patterns.

### Detection Output Storage & Schema
Detections are automatically structured and saved as a CSV dataset:
```text
data/processed/<video_id>/detections.csv
```

The CSV preserves complete frame and bounding box provenance:
| Field | Type | Description |
| :--- | :--- | :--- |
| `video_id` | `str` | Video identifier safely derived from filename |
| `frame_index` | `int` | 0-indexed position in source video stream |
| `extracted_frame_index` | `int` | 1-indexed sequential frame number in extracted frames |
| `timestamp_seconds` | `float` | Exact elapsed time in source video (seconds) |
| `frame_filename` | `str` | Name of the extracted image (`frame_000001.jpg`) |
| `detection_index` | `int` | 0-indexed detection order within this specific frame |
| `box_x1` | `float` | Bounding box top-left $X$ pixel coordinate |
| `box_y1` | `float` | Bounding box top-left $Y$ pixel coordinate |
| `box_x2` | `float` | Bounding box bottom-right $X$ pixel coordinate |
| `box_y2` | `float` | Bounding box bottom-right $Y$ pixel coordinate |
| `confidence` | `float` | Model confidence score ($0.0 \le c \le 1.0$) |
| `class_id` | `int` | COCO class ID (`0` for person) |
| `class_name` | `str` | Class label (`person`) |

### Interactive Detection Visualizer
The Streamlit application provides:
1. **Model Parameter Controls:** Confidence threshold slider ($0.10$ to $1.00$) and one-click batch detection execution.
2. **Detection Summary Card:** Total frames processed, total people detected, average detections per frame, and min/max detection range.
3. **Interactive Frame Visualizer:** Frame selection slider, high-contrast green bounding boxes with confidence labels (`Person 0.84`), side-by-side or tabbed comparison of original vs. annotated frames, and per-frame detection breakdown table.
4. **Export & Download:** Complete `detections.csv` preview table with a one-click CSV download button.

---

## 9. Automated Testing

Run the comprehensive PyTest suite covering video validation, preprocessing, frame sampling, person detection, and Streamlit UI workflows:

```bash
pytest tests/ -v
```

The 40-test automated suite covers:
* `test_video_utils.py` (14 tests): Filename sanitization, path traversal prevention, extension validation, OpenCV decodability, empty/corrupt file rejection, metadata extraction.
* `test_frame_extractor.py` (13 tests): Image validation, color conversion, resizing, chronological timestamp ordering, sampling ratios (1:1, 1:5, 1:10), CSV schema verification, cache handling.
* `test_detector.py` (7 tests): YOLO model initialization, person detection inference on classroom scenes, confidence threshold filtering, bounding box rendering, empty/zero-person frame handling, invalid inputs, and end-to-end batch pipeline execution.
* `test_app.py` (6 tests): Streamlit end-to-end UI integration tests covering initial render, file upload, metric cards, extraction button triggers, extraction summaries, Feature 3 person detection workflows, previews, and corrupted upload handling.

---

## 10. Current Limitations (Features 1, 2 & 3 Scope)

Features 1, 2, and 3 focus exclusively on **Classroom Video Ingestion, Preprocessing, Chronological Frame Extraction, and Spatial Person Detection**.

Current limitations:
* Student tracking across temporal sequences is not yet active (no persistent IDs across frames; reserved for Feature 4).
* Observable classroom behaviour classification is not yet implemented (reserved for Feature 5).
* CNN visual feature extraction is not yet active (Feature 6).
* Temporal sequence modeling (RNN / LSTM / GRU) is not yet active (Feature 7).
* Cloud / cluster distributed processing is not yet enabled.

---

## 11. Future Research Pipeline Roadmap

The subsequent development phases will follow this structured academic pipeline:

```text
Classroom Video Input (Feature 1 — Completed)
   ↓
Frame Extraction & Preprocessing (Feature 2 — Completed)
   ↓
Student Detection (YOLO / Spatial Bounding Boxes) (Feature 3 — Completed)
   ↓
Student Tracking (DeepSORT / ByteTrack Multi-Object Tracking) (Feature 4 — Upcoming)
   ↓
Observable Behaviour Recognition (Spatial Action Analysis) (Feature 5)
   ↓
CNN Visual Feature Extraction (Spatial Representations) (Feature 6)
   ↓
Temporal Sequence Creation (Sliding Window Time Sequences) (Feature 7)
   ↓
Sequence Modeling (RNN / LSTM / GRU) (Feature 8)
   ↓
Observable Behaviour Trajectory Profiling (Feature 9)
   ↓
Teaching Activity Correlation Analysis (Feature 10)
   ↓
Baseline Model Comparison (Feature 11)
   ↓
Ablation Studies & Temporal Error Analysis (Feature 12)
   ↓
Final Interactive Analytics Dashboard (Feature 13)
```
