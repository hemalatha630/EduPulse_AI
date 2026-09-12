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
│   ├── frames/                     # Extracted video frames (future)
│   └── processed/                  # Processed datasets (future)
│
├── models/                         # Trained model weights & checkpoints (future)
├── notebooks/                      # Research & exploratory notebooks
│
├── src/                            # Modular source code
│   ├── __init__.py
│   ├── config.py                   # Central paths, supported types, & constraints
│   ├── video/                      # Feature 1 video module
│   │   ├── __init__.py
│   │   └── video_utils.py          # Video validation, metadata extraction, sanitization
│   ├── preprocessing/              # Frame preprocessing (future)
│   ├── detection/                  # Student detection (future)
│   ├── tracking/                   # Student tracking (future)
│   ├── behaviour/                  # Observable behaviour classification (future)
│   ├── cnn/                        # Spatial visual feature extraction (future)
│   └── temporal/                   # Sequence modeling (RNN/LSTM/GRU) (future)
│
├── results/                        # Evaluation logs and ablation outputs (future)
├── tests/                          # Automated unit and integration tests
│   ├── __init__.py
│   └── test_video_utils.py         # Automated tests for video ingestion
│
├── requirements.txt                # Strictly minimal dependencies for Feature 1
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

## 7. Automated Testing

Run the test suite to verify filename sanitization, extension checking, OpenCV decodability, corrupted file rejection, and metadata calculations:

```bash
pytest tests/ -v
```

---

## 8. Current Limitations (Feature 1 Scope)

Feature 1 focuses exclusively on **Video Ingestion, File Validation, and Metadata Profiling**.

Current limitations:
* Frame extraction and image normalization are not yet executed.
* Student bounding boxes and spatial tracking are not yet computed.
* Behaviour classification and temporal engagement trajectories are not yet active.
* Analysis is local; cloud processing is not yet implemented.

---

## 9. Future Research Pipeline Roadmap

The subsequent development phases will follow this structured academic pipeline:

```text
Classroom Video Input (Feature 1 - Current)
   ↓
Frame Extraction & Preprocessing
   ↓
Student Detection (YOLO / Spatial Bounding Boxes)
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
