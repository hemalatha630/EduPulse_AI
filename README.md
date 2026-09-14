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

## 2. Technology Stack (Features 1 through 10)

* **Programming Language:** Python 3.12+ (supports Python 3.11+)
* **Web Application Framework:** Streamlit
* **Computer Vision & Video Processing:** OpenCV (`opencv-python`)
* **Object Detection & Deep Learning:** Ultralytics YOLO (`ultralytics`), PyTorch (`torch`, `torchvision`)
* **Multi-Object Tracking:** ByteTrack & BoT-SORT (Linear Assignment Problem solver `lap`)
* **Pretrained CNN Visual Backbone:** PyTorch Torchvision ResNet18 (512-dim visual embeddings)
* **Temporal Sequence Generation:** Sliding-window chunking, NumPy 3D arrays, PyTorch `torch.utils.data.Dataset` (`ClassroomSequenceDataset`) and `DataLoader` compatibility
* **Recurrent Sequence Modelling:** PyTorch RNN, LSTM, and GRU temporal classifiers with track-grouped cross-entropy training
* **Trajectory Profiling & Teaching Activity Analysis:** Continuous segment merging, transition calculation, activity distribution accounting, cross-tabulation heatmaps, and Gantt-style timeline visualizations
* **Visualization & Plotting:** Matplotlib (`matplotlib`), Pillow (`Pillow`)
* **Numerical Computing & SVD/PCA:** NumPy
* **Data Structures & Processing:** Pandas
* **Test Suite:** PyTest (153 automated tests, 100% pass rate)

---

## 3. Project Structure

```text
EduPulse_AI/
│
├── app.py                              # Streamlit main application entry point (Features 1-10)
│
├── data/                               # Data storage (git-ignored for student privacy)
│   ├── videos/                         # Uploaded raw classroom videos
│   ├── frames/                         # Extracted and sampled video frames (<video_id>/)
│   └── processed/                      # Processed metadata, tracking, behaviours & CNN outputs (<video_id>/)
│       └── <video_id>/
│           ├── frame_metadata.csv      # Feature 2 chronological frame index
│           ├── detections.csv          # Feature 3 per-frame bounding box coordinates
│           ├── tracks.csv              # Feature 4 persistent temporal track records
│           ├── behaviours.csv          # Feature 5 observable behaviour classifications
│           ├── cnn_features.npy        # Feature 6 raw (N, 512) float32 feature array
│           ├── cnn_features_metadata.csv # Feature 6 spatial-temporal metadata mapping
│           ├── temporal_sequences.npy  # Feature 7 3D float32 sequence tensor (N, L, D)
│           ├── temporal_sequences_metadata.csv # Feature 7 sequence ledger & window mapping
│           └── teaching_activity_segments.csv  # Feature 10 validated instructional segment annotations
│
├── models/                             # Model weights directory
│   ├── yolov8n.pt                      # Pretrained YOLOv8n detector (~6.2 MB)
│   └── temporal/                       # Feature 8 trained recurrent model checkpoints
│       ├── rnn_best.pt                 # Trained Vanilla RNN weights
│       ├── lstm_best.pt                # Trained LSTM weights
│       └── gru_best.pt                 # Trained GRU weights
│
├── notebooks/                          # Research & exploratory notebooks
│
├── results/                            # Experimental outputs and evaluation ledgers
│   ├── temporal/                       # Feature 8 recurrent training histories and predictions
│   ├── trajectories/                   # Feature 9 observable behaviour trajectories
│   └── teaching_activity/              # Feature 10 teaching activity summaries & distributions (<video_id>/)
│       └── <video_id>/
│           ├── activity_behaviour_summary_<model>.csv
│           ├── activity_behaviour_distributions_<model>.csv
│           └── activity_transitions_<model>.csv
│
├── src/                                # Modular source code
│   ├── __init__.py
│   ├── config.py                       # Central paths, sampling defaults, activity & colour palettes
│   ├── video/                          # Video ingestion & frame extraction (Feature 1 & 2)
│   ├── preprocessing/                  # Image validation, color handling, & resizing
│   ├── detection/                      # Student / Person Detection (Feature 3)
│   ├── tracking/                       # Student / Person Tracking (Feature 4)
│   ├── behaviour/                      # Observable behaviour classification (Feature 5)
│   ├── cnn/                            # CNN Visual Feature Extraction (Feature 6)
│   ├── temporal/                       # Temporal sequence creation & recurrent modeling (Feature 7 & 8)
│   ├── trajectory/                     # Observable behaviour trajectory extraction (Feature 9)
│   └── activity/                       # Teaching Activity Analysis & Distributions (Feature 10)
│       ├── __init__.py                 # Public package export
│       ├── activity_labels.py          # Canonical activity constants, descriptions, & colors
│       ├── manager.py                  # TeachingActivitySegment dataclass, validation, & disk I/O
│       ├── analyzer.py                 # Mapping, distribution calculation, summary, ties & transitions
│       └── visualization.py            # Timelines, grouped bars, cross-tabulation heatmaps, & track charts
│
├── tests/                              # Automated unit and integration tests (153 tests)
│   ├── test_app.py                     # Streamlit UI integration tests (Features 1-10)
│   ├── test_video_utils.py             # Video validation unit tests
│   ├── test_frame_extractor.py         # Frame extraction, sampling, & preprocessor tests
│   ├── test_detector.py                # YOLO person detection & annotation unit tests
│   ├── test_tracker.py                 # Multi-object tracking, ID consistency, & trajectory tests
│   ├── test_behaviour.py               # Observable behaviour recognition unit tests
│   ├── test_cnn.py                     # CNN ResNet18 loading, batch extraction, & PCA tests
│   ├── test_temporal.py                # Temporal sequence generation & PyTorch Dataset tests
│   ├── test_temporal_models.py         # Recurrent model training, evaluation, & inference tests
│   ├── test_trajectory.py              # Observable behaviour trajectory extraction & visualization tests
│   └── test_activity.py                # Feature 10 segment validation, distributions, & visualization tests
│
├── requirements.txt                    # Core dependencies
├── .gitignore                          # Git ignore rules for video data & environment
└── README.md                           # Comprehensive project documentation
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

## 9. Feature 4 — Student / Person Tracking

Feature 4 bridges frame-level spatial detection and future temporal sequence modeling by connecting individual person detections across consecutive video frames into persistent, anonymous **Track IDs**.

### What Is Multi-Object Tracking & Why Is It Needed?
* **Detection (Feature 3):** Identifies *where* people are located in isolated frames without any memory of previous frames.
* **Tracking (Feature 4):** Identifies *which* detection in frame $t$ corresponds to the *same* physical person in frame $t+1$, preserving continuity across time.
* **Why Tracking Is Essential:** Analyzing temporal behaviour trajectories (Feature 5+) requires linking observations to the same individual over a time window rather than treating each frame as an unrelated collection of people.

### Selected Tracking Algorithms
* **ByteTrack (Default & Recommended):** High-speed association algorithm that matches both high-confidence and low-confidence detection boxes using Kalman filter state predictions and IoU matching. This retains tracks even when a student is momentarily occluded by a peer, laptop, or desk.
* **BoT-SORT (Supported Alternative):** Integrates camera motion compensation (GMC) and enhanced Kalman filtering for dynamic camera setups.

### Anonymous Track IDs vs. Real Identities
> [!IMPORTANT]
> **Feature 4 maintains anonymous temporary Track IDs for detected people across video frames. These IDs DO NOT represent real student identities.**
> - Track IDs (e.g. `ID: 1`, `ID: 2`) are arbitrary integers generated solely for mathematical continuity.
> - The system strictly **DOES NOT** store student names, roll numbers, or university IDs.
> - Facial recognition, identity classification, demographic estimation (age/gender), and subjective emotional profiling are strictly excluded.

### Classroom Tracking Challenges & Occlusion Handling
Classroom environments present unique computer vision challenges:
* **Dense Seating & Mutual Occlusion:** Students in adjacent lecture rows frequently overlap in camera perspective. ByteTrack's two-stage matching recovers occluded students without immediately terminating tracks.
* **ID Switches:** If a student is obscured behind a standing peer or moves completely out of view for several seconds, the tracker may instantiate a new Track ID upon re-detection. This is a known, expected characteristic of visual tracking.
* **Spatial Centroid Paths:** Tracking records image-plane bounding box centers $(center\_x, center\_y)$. These represent physical motion across frames, **not** cognitive engagement, attention, or comprehension.

### Tracking Output Storage & Schema (`tracks.csv`)
Tracking outputs are saved in structured tabular format:
```text
data/processed/<video_id>/tracks.csv
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `video_id` | `str` | Video identifier derived from filename |
| `frame_id` | `int` | 0-indexed position in source video stream |
| `extracted_frame_index` | `int` | 1-indexed sequential frame number in extracted frames |
| `timestamp_seconds` | `float` | Exact elapsed time in source video (seconds) |
| `frame_filename` | `str` | Associated image filename (`frame_000001.jpg`) |
| `track_id` | `int` | Anonymous temporary tracking integer identifier |
| `class_id` | `int` | COCO class ID (`0` for person) |
| `class_name` | `str` | Class label (`person`) |
| `confidence` | `float` | Detection confidence score ($0.0 \le c \le 1.0$) |
| `x1` | `float` | Bounding box top-left $X$ coordinate (pixels) |
| `y1` | `float` | Bounding box top-left $Y$ coordinate (pixels) |
| `x2` | `float` | Bounding box bottom-right $X$ coordinate (pixels) |
| `y2` | `float` | Bounding box bottom-right $Y$ coordinate (pixels) |
| `center_x` | `float` | Bounding box center point horizontal coordinate $(x_1 + x_2)/2$ |
| `center_y` | `float` | Bounding box center point vertical coordinate $(y_1 + y_2)/2$ |

### Interactive Tracking UI Capabilities
1. **Configurable Controls:** Tracker algorithm selector (ByteTrack / BoT-SORT), confidence slider ($0.10 - 1.00$), sequence range selector, and trajectory trail toggle.
2. **Tracking Summary Cards:** Total frames processed, unique tracks, average active tracks per frame, and longest continuous track duration.
3. **Sequential Tracking Visualizer:**
   - Single Frame View: Persistent colored bounding boxes, Track ID badges, centroid dots, and trajectory trails.
   - Consecutive Comparison View: Side-by-side comparison of Frame $N-1$ and Frame $N$ with highlighted persisting track IDs.
4. **Spatial Trajectory Map:** 2D centroid movement scatter/line plot displaying paths over time across classroom coordinates.
5. **Dataset Export:** Structured preview table and one-click `tracks.csv` download.

---

## 10. Feature 5 — Observable Behaviour Recognition

Feature 5 recognizes **observable learning-related behaviours** for each tracked person in classroom video frames. It consumes persistent bounding boxes from Feature 4 (`tracks.csv`), safely crops and normalizes person regions, and assigns one of six canonical behaviour categories (or an uncertain state) based strictly on visible evidence.

### Why Observable Behaviour Recognition?
In educational research, learning engagement begins with observable actions. Before applying temporal sequence models (RNN/LSTM/GRU in Features 7–8), the system must extract reliable, objective, frame-level classifications of what visible actions a student is performing over time.

### The Six Target Observable Behaviour Classes
The system classifies exclusively observable physical evidence:

1. **Looking toward the instructional activity:** Body or head oriented toward the instructor, presentation screen, or front blackboard area.
2. **Reading/writing:** Head angled downward toward a notebook, book, or paper surface with visible hand or arm posture associated with writing or reading.
3. **Interacting with peers:** Head or upper body oriented toward an adjacent or nearby student with collaborative or conversational proximity.
4. **Looking away:** Head or torso oriented substantially away from instructional activity toward windows, doors, or lateral periphery.
5. **Mobile-device activity:** Visible handheld mobile phone or tablet, or gaze directed downward into a handheld device in the lap or desk region.
6. **Head-down behaviour:** Head resting directly on desk surface, arms folded under head, or face hidden downward without reading/writing movement.
7. **Unknown / Uncertain:** Assigned when visual evidence is ambiguous, severe occlusion occurs, crops are blurry/low-resolution, or classifier confidence falls below threshold.

### Strict Research & Ethical Boundaries
> [!IMPORTANT]
> **Feature 5 classifies strictly OBSERVABLE behaviour only. It DOES NOT infer internal mental, emotional, or cognitive states.**
> - **Zero Emotional/Cognitive Labels:** Labels such as *bored*, *motivated*, *sad*, *happy*, *intelligent*, *confused*, *mentally engaged*, or *understanding* are strictly forbidden.
> - **Observable Action $\ne$ Mental State:**
>   - *Looking Away* does NOT prove a student is bored or disengaged.
>   - *Looking toward instruction* does NOT prove a student understands the material.
>   - *Mobile-device activity* describes an observed object and posture, not student motivation.
> - **Confidence Interpretation:** Classification confidence measures the model's certainty regarding the visual category, **not** the student's degree of engagement or attention.
> - **Privacy Preservation:** Anonymous Track IDs from Feature 4 are retained; facial recognition, biometric matching, and demographic profiling are strictly excluded.

### Modular Architecture: Situation A vs. Situation B
To ensure academic honesty and reproducibility, Feature 5 supports two operating situations:

* **Situation A (Trained PyTorch Model):** If trained neural network weights exist (e.g. `models/behaviour_classifier.pt`), the classifier loads the model and performs forward inference on normalized crops.
* **Situation B (Prototype / Baseline Heuristic — Active):** Because labelled classroom behaviour datasets are proprietary or in collection, the system operates in a transparent **Prototype / Baseline Heuristic Pipeline**. It evaluates measurable visual cues:
  - **Peer Proximity:** Euclidean distance between tracked centroids within the same frame to detect collaborative alignment ($d < 1.35 \times \text{width}$).
  - **Upper Body / Head Skew:** Horizontal center of mass in head region to detect lateral gaze (*Looking Away*).
  - **Lap/Desk Surface Texture:** Laplacian and Canny edge density in lower crop region to distinguish active *Reading/Writing* from *Head-Down* postures.
  - **High-Contrast Lap Objects:** Luminance ratio in hand region to flag *Mobile-Device Activity*.
  - **Threshold Rejection:** Low confidence, blur, or sub-minimum crops default cleanly to *Unknown / Uncertain*.
  - **No Fake Accuracy Claims:** The system explicitly labels its active mode in the UI and never claims fabricated ML benchmarks.

### Person Crop Preprocessing
Located in `src/behaviour/preprocessing.py`:
* **Safe Coordinate Clipping:** Clips bounding box coordinates strictly to frame dimensions `[0, W]` and `[0, H]`.
* **Geometry Validation:** Rejects inverted, zero-area, or crops smaller than $20 \times 30$ pixels without crashing.
* **Standard Resizing:** Resizes crops to $224 \times 224$ pixels via bilinear interpolation.
* **PyTorch Normalization:** Outputs standard ImageNet-normalized tensors `(1, 3, 224, 224)` ready for CNN feature extraction in Feature 6.

### Behaviour Output Storage & Schema (`behaviours.csv`)
Saved to:
```text
data/processed/<video_id>/behaviours.csv
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `video_id` | `str` | Video identifier derived from filename |
| `frame_id` | `int` | 0-indexed position in source video stream |
| `extracted_frame_index` | `int` | 1-indexed sequential frame number in extracted frames |
| `timestamp_seconds` | `float` | Elapsed time in source video (seconds) |
| `frame_filename` | `str` | Name of associated image (`frame_000001.jpg`) |
| `track_id` | `int` | Anonymous integer Track ID from Feature 4 |
| `behaviour_class` | `str` | Classified observable behaviour category |
| `confidence` | `float` | Classification confidence score ($0.0 \le c \le 1.0$) |
| `x1` | `float` | Bounding box top-left $X$ coordinate |
| `y1` | `float` | Bounding box top-left $Y$ coordinate |
| `x2` | `float` | Bounding box bottom-right $X$ coordinate |
| `y2` | `float` | Bounding box bottom-right $Y$ coordinate |
| `visual_evidence` | `str` | Brief explanation of visual cues supporting the classification |

### Interactive Streamlit UI Capabilities
1. **Classifier Mode Indicator:** Displays active inference mode (*Prototype / Baseline Heuristic* or *PyTorch Neural Model*).
2. **Confidence Threshold Slider:** Filter predictions below $\tau$ into *Unknown / Uncertain* to maintain scientific rigor.
3. **Summary Metric Cards:** Total observations, dominant observable behaviour, unique tracks covered, and unknown/uncertain count.
4. **Distribution Chart:** Horizontal bar chart displaying observation frequencies across all behaviour classes with distinct color mapping.
5. **Frame Visualizer:** Interactive slider displaying frames with colored bounding boxes, Track IDs, and behaviour badges (`ID 1: Interacting with peers | Conf: 0.77`).
6. **Track-Level Chronological Sequence Inspector:** Inspect individual student behaviour timelines over time (with prominent disclaimer that this is simple observation logging, not temporal modeling).
7. **Dataset Download:** Complete `behaviours.csv` preview table with a one-click CSV download button.

---

## 11. CNN Visual Feature Extraction (Feature 6)

### Purpose & Rationale
Feature 6 bridges spatial student tracking and future temporal sequence modeling by converting raw person image crops into compact, fixed-length **512-dimensional visual feature vectors**.

```text
Person Image Crop (224x224 RGB)
              ↓
    Pretrained ResNet18 CNN
  (Final FC Layer → Identity)
              ↓
  Fixed 512-D Visual Embedding
              ↓
    Decoupled Storage Architecture
  (.npy Binary Array + .csv Metadata)
```

### Pretrained CNN Model Selection: ResNet18
The system employs **ResNet18** (`torchvision.models.resnet18` with ImageNet-1K pretrained weights) as a frozen visual feature extractor.

**Why ResNet18 was selected:**
1. **Fixed Feature Dimension (512D):** By replacing the final classification layer (`model.fc = nn.Identity()`), ResNet18 naturally outputs a 512-dimensional vector from its global average pooling layer.
2. **Computational Efficiency & Low Latency:** With ~11.7 million parameters (compared to 25M+ for ResNet50 or 86M+ for ViT), ResNet18 executes efficiently on standard classroom computer hardware and laptops without requiring dedicated GPUs.
3. **Scientific Reproducibility:** ResNet18 is a widely recognized standard baseline in computer vision and educational video analytics literature.
4. **Hardware Agnostic:** Automatically utilizes CUDA acceleration when a compatible GPU is available; gracefully and safely executes on CPU when running in standard academic environments.

> **Important Conceptual Distinction:**
> Feature 6 uses a pretrained CNN as a visual feature extractor for tracked person regions. The extracted feature vectors represent visual information and do not directly represent internal student mental states.
> 
> Temporal modelling using RNN/LSTM/GRU will be implemented in later features.

### Person Crop Preprocessing & Batch Extraction
For every tracked student in every extracted frame:
1. **Boundary Clipping:** Bounding box coordinates $(x_1, y_1, x_2, y_2)$ are safely clipped to frame boundaries $[0, W]$ and $[0, H]$.
2. **Geometry Validation:** Crops with width $< 20\text{px}$ or height $< 30\text{px}$ are safely filtered and recorded as skipped crops.
3. **Bilinear Resizing:** Crops are resized to the CNN input standard $224 \times 224$ pixels.
4. **ImageNet Normalization:** Pixel intensities are converted to $[0.0, 1.0]$ and standardized:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
5. **Batched Forward Pass:** Crops within a frame are assembled into mini-batches (default size: 16) and passed through the model under `torch.no_grad()` evaluation mode for high throughput.

### Decoupled Storage Architecture
To prevent bloating tabular CSV files with 512 numerical columns, features are stored using a clean decoupled design:

```text
data/processed/<video_id>/
├── cnn_features.npy            # Raw float32 binary matrix of shape (N, 512)
└── cnn_features_metadata.csv   # Structured mapping table linking rows 0..N-1
```

#### Metadata Schema (`cnn_features_metadata.csv`)
| Column | Type | Description |
| :--- | :--- | :--- |
| `video_id` | `str` | Video identifier stem |
| `frame_id` | `int` | Original video frame index |
| `extracted_frame_index` | `int` | Chronological extracted frame number |
| `timestamp_seconds` | `float` | Elapsed playback time in seconds |
| `frame_filename` | `str` | Frame image file name (`frame_000001.jpg`) |
| `track_id` | `int` | Persistent student Track ID from Feature 4 |
| `confidence` | `float` | Tracking detection confidence score |
| `x1, y1, x2, y2` | `float` | Spatial bounding box coordinates |
| `feature_index` | `int` | Exact 0-indexed row position in `cnn_features.npy` |
| `feature_path` | `str` | Relative reference to feature binary (`cnn_features.npy`) |
| `behaviour_class` | `str` | Linked observable behaviour from Feature 5 |
| `behaviour_confidence` | `float` | Classification confidence of linked behaviour |

### Interactive Streamlit Interface
* **CNN Hardware & Model Controls:** Live detection of compute hardware (`CPU` or `CUDA`), feature dimension readout (`512D`), batch size selector.
* **Flexible Frame Range Processing:** Choose between *All extracted frames*, *Sample frames (first N)*, or *Custom frame range*.
* **Visual Feature Inspector:** Select any processed frame and tracked student to view their $224 \times 224$ visual crop alongside their 512-D vector preview and statistical properties (L2 norm, mean, std, min, max).
* **2D PCA Feature Space Distribution:** Interactive 2D projection computed via pure NumPy SVD (zero scikit-learn dependency), visualizable by Track ID or Observable Behaviour.
* **Artifact Downloads:** One-click download buttons for both `cnn_features_metadata.csv` and raw binary `cnn_features.npy`.

---

## 12. Feature 7 — Temporal Sequence Creation

Feature 7 transforms individual frame-level CNN visual embeddings (Feature 6) into **chronologically ordered, fixed-length sliding-window temporal sequences** grouped strictly by tracked student (`track_id`). It forms the essential data preparation layer bridging static computer vision and downstream temporal sequence modeling (Feature 8).

```text
Feature 6 CNN Vectors (N_frames, 512) + Tracking (track_id)
                      ↓
           Track Partitioning & Isolation
                      ↓
    Chronological Sorting (extracted_frame_index)
                      ↓
     Tracking Gap Detection & Segment Splitting
                      ↓
       Sliding-Window Chunking (L=10, S=2)
                      ↓
      Temporal Sequences Tensor (N, L, 512)
                      +
  Spatial-Temporal Sequence Metadata (CSV ledger)
                      ↓
  PyTorch ClassroomSequenceDataset & DataLoader (Feature 8 Ready)
```

### Academic & Engineering Rationale
* **Separation of Concerns:** Data organization is strictly decoupled from model training/inference. Sequence creation structures temporal representations without assuming a specific classifier architecture.
* **Track-Wise Isolation:** Each sequence contains observations from exactly one student track. Observations from different individuals are **never mixed**, preventing artificial transition artifacts.
* **Preservation of Chronological Flow:** Within each track, observations are sorted by extracted frame index and video timestamp to ensure strictly increasing time steps ($t_1 < t_2 < \dots < t_L$).
* **Sliding Window Formulation:** Given a continuous track segment of $M$ observations, window length $L$, and stride $S$, the number of generated sequences is:
  $$\text{Num Sequences} = \left\lfloor \frac{M - L}{S} \right\rfloor + 1$$
* **Gap Splitting Policy:** If the tracking gap between consecutive observations exceeds tolerance ($G$ frames, default 2), the track is split into continuous sub-segments to prevent unobserved jumps.
* **Short Track Skipping:** Track segments with fewer than $L$ observations are safely skipped and reported in the metrics summary, avoiding distorted zero-padded windows.
* **Observable Behaviour Alignment:** Each sequence window is annotated with its dominant observable behaviour (statistical mode), mean classification confidence, transition chain string (`A -> B -> C`), and explicit list of frame indices and timestamps.
* **Prevention of Data Leakage:** Because all sequences maintain explicit `track_id` attribution, future train/validation/test splits can be performed strictly at the student level rather than randomly splitting overlapping windows.

### Artifacts & Decoupled Storage (`data/processed/<video_id>/`)

1. **`temporal_sequences.npy`**: 3D float32 NumPy array of shape $(N_\text{sequences}, L, D)$ storing the raw numerical visual sequences.
2. **`temporal_sequences_metadata.csv`**: Comprehensive tabular ledger mapping each sequence index to its spatial-temporal parameters:

| Column | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | `int` | Sequential 0-indexed sequence identifier matching the first dimension of `.npy` |
| `video_id` | `str` | Video identifier stem |
| `track_id` | `int` | Persistent student Track ID from Feature 4 |
| `start_frame_id` | `int` | Original video frame index at sequence start ($t_1$) |
| `end_frame_id` | `int` | Original video frame index at sequence end ($t_L$) |
| `start_extracted_frame_index` | `int` | Chronological extracted frame number at window start |
| `end_extracted_frame_index` | `int` | Chronological extracted frame number at window end |
| `start_timestamp_seconds` | `float` | Video playback timestamp at window start |
| `end_timestamp_seconds` | `float` | Video playback timestamp at window end |
| `duration_seconds` | `float` | Total elapsed duration of the sequence window in seconds |
| `sequence_length` | `int` | Number of time steps ($L$, default 10) |
| `feature_dimension` | `int` | Visual embedding dimensionality ($D=512$) |
| `dominant_behaviour` | `str` | Most frequent observable behaviour class within the sequence window |
| `mean_behaviour_confidence` | `float` | Average classification confidence of behaviours across the window |
| `behaviour_sequence` | `str` | Full transition chain across window steps (`Listening -> Writing`) |
| `frame_indices` | `str (JSON)` | JSON array of exact extracted frame numbers for all $L$ steps |
| `frame_timestamps` | `str (JSON)` | JSON array of exact playback timestamps for all $L$ steps |

### PyTorch Integration (`ClassroomSequenceDataset`)
A native PyTorch `Dataset` adapter is provided in `src.temporal`:

```python
from torch.utils.data import DataLoader
from src.temporal import ClassroomSequenceDataset
import numpy as np
import pandas as pd

# Load decoupled artifacts
sequences = np.load("data/processed/video_id/temporal_sequences.npy")  # (N, 10, 512)
metadata = pd.read_csv("data/processed/video_id/temporal_sequences_metadata.csv")

# Create PyTorch Dataset & DataLoader
dataset = ClassroomSequenceDataset(sequences=sequences, metadata_df=metadata)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Ready for Feature 8 sequence models (RNN / LSTM / GRU)
for batch_tensors, batch_meta in dataloader:
    # batch_tensors shape: torch.Size([8, 10, 512])
    pass
```

### Interactive Streamlit Interface
* **Configurable Controls:** Sliders for sequence length $L$ (3–30), stride $S$ (1–10), and max frame gap tolerance $G$ (1–10).
* **Summary Metrics:** Total sequences created, unique student tracks covered, short tracks skipped, PyTorch tensor shape $(N, L, D)$, and average duration.
* **Interactive Sequence Inspector:** Dropdown to select any sequence and view its Track ID, frame window, video timestamps, dominant behaviour, mean confidence, and transition chain.
* **Sequence Timeline Visualization:** Visual plot showing the visual embedding L2 norm progression across time steps with aligned observable behaviour markers.
* **Track Temporal Coverage Chart:** Gantt-style timeline chart illustrating the coverage windows of temporal sequences across all active student tracks.
* **PyTorch Code Snippet:** Live copy-pasteable PyTorch DataLoader code block populated with actual tensor dimensions.
* **Artifact Downloads:** Direct download buttons for `temporal_sequences_metadata.csv` and binary `temporal_sequences.npy`.

---

---

## 13. Feature 8 — RNN / LSTM / GRU Temporal Modelling

Feature 8 introduces recurrent neural architectures (**Vanilla RNN**, **LSTM**, and **GRU**) to model multi-frame temporal dependencies from sequential visual embeddings (Feature 7) and classify observable classroom learning behaviours.

```text
Temporal Sequences (N, 10, 512)
              ↓
  Track-Grouped Splitting (Zero Leakage: Train 70%, Val 15%, Test 15%)
              ↓
  Balanced Class Weighting (Computed Strictly on Training Set)
              ↓
  Fair Comparison Training Engine (Identical Optimizer, Loss, LR, Patience)
       ┌──────────────┼──────────────┐
  Vanilla RNN        LSTM           GRU
  (nn.RNN)         (nn.LSTM)      (nn.GRU)
       └──────────────┼──────────────┘
              ↓
  Unseen Test Set Evaluation (Accuracy, Macro F1, Weighted F1, Confusion Matrix)
              ↓
  Inference Engine (Single-Sequence Inspector + Batch Predictions CSV)
```

### Academic & Pedagogical Rationale
* **Beyond Static Snapshots:** Single-frame CNN embeddings capture momentary visual appearances but cannot disambiguate temporal phenomena, such as distinguishing brief momentary glances from sustained instructional engagement or tracking transitions between writing and peer interaction.
* **Recurrent Dynamics:** Recurrent neural networks maintain an evolving internal hidden state $h_t$ over time steps $t=1 \dots L$, modeling chronological context:
  $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
* **Strict Observable Scope:** Recurrent classifications are strictly confined to the 6 target observable classroom behaviours. In accordance with ethical computer vision standards, no internal mental, cognitive, or emotional states are inferred.

### Data Leakage Prevention Protocol
* **The Overlapping Window Dilemma:** Because temporal sequences are extracted via overlapping sliding windows (stride $S < L$), adjacent windows share identical video frames. A naive random train/test split would place identical video frames from the same student in both train and test partitions, causing severe data leakage and artificially inflated accuracy.
* **Track-Grouped Splitting:** EduPulse AI partitions data strictly by **Track ID** (`track_id`). All overlapping sequence windows belonging to a given student track are assigned exclusively to either the Train, Validation, or Test set.
* **Training-Only Class Weighting:** Class imbalance weights are calculated strictly from the training partition:
  $$w_c = \frac{N_\text{train}}{|C_\text{present}| \cdot N_{c, \text{train}}}$$
  The validation and test partitions are never accessed during weight calculation to prevent distribution leakage.

### Model Architectures & Fair Comparison Protocol
All three recurrent classifiers share identical input dimensions ($D=512$), output dimensions ($K=6$ classes), classification heads, dropout, optimizer (Adam), learning rate ($1 \times 10^{-3}$), and early stopping patience (5 epochs on validation loss):

1. **Vanilla RNN (`RNNClassifier`):**
   * PyTorch `nn.RNN` recurrent layer (`batch_first=True`)
   * Fully connected projection head with dropout
   * Fast, lightweight baseline (~37K parameters at hidden size 64)
2. **Long Short-Term Memory (`LSTMClassifier`):**
   * PyTorch `nn.LSTM` with cell state $c_t$ and three gating mechanisms (forget $f_t$, input $i_t$, output $o_t$)
   * Mitigates vanishing/exploding gradients across long multi-frame sequences (~148K parameters at hidden size 64)
3. **Gated Recurrent Unit (`GRUClassifier`):**
   * PyTorch `nn.GRU` with update $z_t$ and reset $r_t$ gates
   * Combines cell state and hidden state for computational efficiency (~111K parameters at hidden size 64)

### Test Evaluation & Diagnostics
Models are evaluated strictly on the unseen test tracks using comprehensive metrics:
* **Overall Metrics:** Accuracy, Macro-averaged F1, Weighted F1, Macro Precision, Macro Recall.
* **Per-Class Breakdown:** Precision, Recall, F1-Score, and Support across each observable behaviour class.
* **6×6 Confusion Matrix:** Annotated heatmap visualising true vs predicted behaviour distributions.
* **Fair Comparison Table & Chart:** Grouped bar chart and tabular benchmark comparing all three models side-by-side.

### Inference & Decoupled Storage
* **Single-Sequence Inspector:** Interactive UI component allowing users to select any test sequence, run live inference, inspect predicted vs ground truth labels, and view probability distributions across all 6 classes.
* **Batch Predictions Ledger (`results/temporal/predictions.csv`):** Row-by-row prediction records with Track ID, sequence window timestamps, true labels, predicted labels, and confidence scores.
* **Model Checkpoints (`models/temporal/`):** Best checkpoint weights (`rnn_best.pt`, `lstm_best.pt`, `gru_best.pt`) saved with full model configuration metadata.

---

---

## 14. Feature 9 — Observable Behaviour Trajectory

Feature 9 visualizes and analyzes how observable learning-related behaviours change over time for anonymous tracked individuals across the classroom video using predictions generated by trained recurrent temporal models (Vanilla RNN, LSTM, and GRU).

```text
Feature 8 Model Checkpoints (RNN / LSTM / GRU) + Feature 7 Sequences (N, 10, 512)
                              ↓
              Zero-Retraining Inference Engine
   (Batched torch.no_grad() forward pass across all video sequences)
                              ↓
       Predictions Cache: results/trajectories/<video_id>/predictions_<model>.csv
                              ↓
  ┌───────────────────────────────────────────────────────────┐
  │                 Trajectory Analysis Engine                │
  │  • Track-Wise Isolation & Time-Range Window Filtering     │
  │  • Tracking Gap & Discontinuity Detection (Occlusions)    │
  │  • Consecutive Identical Prediction Merging (Segments)    │
  │  • Chronological Transition Accounting (A → B shifts)     │
  │  • Observed Durations (seconds) & Percentage of Time      │
  │  • MM:SS Video Playback Timestamp Formatting              │
  └───────────────────────────────────────────────────────────┘
                              ↓
  ┌───────────────────────────────────────────────────────────┐
  │              Visualizations & Diagnostic Views            │
  │  1. Categorical Gantt-Style Timeline (Discrete Bands)     │
  │  2. Observed Duration & Percentage Distribution Chart     │
  │  3. Observable Transitions Frequency Chart                │
  │  4. Multi-Model Architecture Comparison (Stacked Rows)    │
  │  5. Classroom Multi-Track Overview (All Students)         │
  └───────────────────────────────────────────────────────────┘
                              ↓
       Consolidated Export: results/trajectories/<video_id>/behaviour_trajectories.csv
```

### Academic & Pedagogical Rationale
* **Observable Physical Behaviour Only:** Trajectories chart observable actions (looking toward instructional activity, reading/writing, peer interaction, looking away, mobile-device activity, head-down posture). In accordance with strict ethical computer vision principles, this system does **not** estimate internal cognitive states, comprehension, attention scores, boredom, or motivation.
* **Discrete Categorical Representation:** Learning behaviours are distinct qualitative actions, not ordinal numbers. Plotting them on continuous numerical line charts ($1 \dots 6$) would imply false numerical distance between categories (e.g. suggesting "Reading/writing" is halfway between "Looking toward instruction" and "Interacting with peers"). Feature 9 strictly utilizes **Gantt-style horizontal categorical bands**.
* **Preservation of Discontinuities:** Tracking gaps caused by student occlusion or movement are explicitly detected and rendered with hatched patterns (`//`), preventing false interpolation across missing observations.
* **Anonymous Individual Tracking:** All trajectories are indexed by persistent numerical Track IDs (`track_id`) without demographic identifiers or facial recognition.

### Zero-Retraining Inference Architecture
Feature 9 is strictly an inference and diagnostic layer. It **never retrains** any neural network:
* Automatically loads saved weights (`models/temporal/{rnn,lstm,gru}_best.pt`) from Feature 8.
* Runs a batched `torch.no_grad()` forward pass across all sequences in `temporal_sequences.npy`.
* Caches sequence-level predictions to `results/trajectories/<video_id>/predictions_<model>.csv`.
* Reuses cached predictions for instantaneous UI rendering and interactive exploration.

### Trajectory Extraction Engine (`src/trajectory/extractor.py`)
1. **Track Isolation & Time-Range Filtering (`extract_track_trajectory`):** Filters predictions strictly for an individual student track within user-selected playback bounds ($t_\text{start} \le t \le t_\text{end}$).
2. **Tracking Gap Detection (`detect_tracking_gaps`):** Detects discontinuous observation gaps where $\Delta t = t_{\text{start}, i+1} - t_{\text{end}, i} > \tau_\text{gap}$.
3. **Continuous Segment Merging (`merge_behaviour_segments`):** Merges consecutive identical predictions into continuous `BehaviourSegment` intervals with start/end frames, timestamps, duration, and mean confidence, strictly halting at tracking gaps.
4. **Transition Frequency Accounting (`calculate_behaviour_transitions`):** Logs every chronological boundary where the observed behaviour shifted ($B_i \ne B_{i+1}$), recording transition direction, timestamp, and frequency.
5. **Durations & Relative Share (`calculate_behaviour_durations_and_distribution`):** Computes total observed seconds and percentage of observed time per behaviour category.
6. **Timestamp Formatting (`format_timestamp_mmss`):** Standardizes all video playback timestamps into `MM:SS` strings (e.g. $75.5\text{s} \to \text{"01:15"}$).

### Visualizations & Diagnostic Views
1. **Primary Categorical Timeline:** Gantt-style horizontal bar chart showing continuous behaviour segments in canonical pedagogical colors, with segment duration labels and hatched tracking gaps.
2. **Observed Duration & Percentage Share:** Horizontal bar chart displaying cumulative seconds and percentage of observation window for each detected behaviour.
3. **Observable Behaviour Transitions:** Bar chart illustrating direct transition counts between distinct behaviours (e.g. *Looking toward instruction → Reading/writing*).
4. **Recurrent Architecture Comparison:** Multi-row stacked timeline comparing predictions from Vanilla RNN, LSTM, and GRU side-by-side for the same student track.
5. **Classroom Multi-Track Overview:** Stacked timeline view presenting observable behaviour trajectories for all detected students simultaneously.

### Decoupled Storage & CSV Export (`results/trajectories/<video_id>/`)
* **`predictions_<model>.csv`:** Sequence-level prediction records with track ID, sequence window timestamps, predicted labels, and confidence scores.
* **`behaviour_trajectories.csv`:** Consolidated trajectory export mapping each sequence to its parent continuous segment ID, segment start/end timestamps, and duration.

---

---

## 15. Feature 10 — Teaching Activity Analysis

Feature 10 extends observable behaviour trajectory profiling into instructional context analysis. It answers the fundamental pedagogical research question: **How do observable learning-related student behaviours differ across different classroom teaching activities?**

### Pedagogical Foundations & Strict Scientific Boundary
* **Instructional Setting Description**: Teaching activity classification describes the *instructional structure* of the lesson (e.g., lecture, discussion, problem-solving, presentation).
* **No Psychological State Inferences**: Teaching activities and observable behaviours do **not** measure internal student mental states, motivation, concentration, comprehension, boredom, intelligence, or emotional states.
* **Provenance Attribution**: Activity labels originate strictly from validated manual segment annotations or curriculum metadata (`Activity source: Manual annotation`). No AI hallucination or fabricated activity labels are used.

### Canonical Target Teaching Activities
1. **Lecture (`#1F4E79` — Deep Blue):** Instructor-led instructional delivery where information, concepts, and explanations are presented to the whole classroom.
2. **Discussion (`#2E7D32` — Emerald Green):** Interactive verbal exchange involving student-to-student or teacher-to-student dialogues, Q&A, or structured collaborative conversations.
3. **Problem-solving (`#6A1B9A` — Amethyst Purple):** Task-centered active learning where students work individually or collaboratively on exercises, worksheets, or laboratory tasks.
4. **Presentation (`#D35400` — Rust Orange):** Formal or informal delivery where designated students or guest speakers present projects, solutions, or demonstrations to peers.

### Core Computational Engine (`src/activity/`)
1. **Teaching Activity Segment Management (`src/activity/manager.py`):**
   * `TeachingActivitySegment` dataclass with `MM:SS` formatted accessors.
   * `validate_activity_segments`: Strict temporal validation verifying non-negative timestamps, start < end, end within video duration, canonical activity types, and rigorous overlap detection ($\Delta t \le 10^{-4}\text{s}$).
   * `load_teaching_activity_segments` / `save_teaching_activity_segments`: Disk persistence to `data/processed/<video_id>/teaching_activity_segments.csv`.
   * `create_default_demo_segments`: Automatic partitioned demo intervals for demonstration videos.
2. **Activity-Behaviour Mapping & Distribution Accounting (`src/activity/analyzer.py`):**
   * `map_predictions_to_activities`: Assigns activity segments to recurrent predictions based on temporal occurrence without re-training models (`torch.no_grad()`).
   * `calculate_activity_behaviour_distributions`: Computes exact observed durations (seconds) and relative percentage shares per behaviour within each activity, strictly avoiding double-counting across overlapping sequence windows.
   * `generate_activity_summary_table`: Aggregates activity duration, active track counts, total observed time, dominant behaviour, dominant percentage share, and explicit tie detection (`is_tie = True`).
   * `calculate_activity_transitions`: Chronological behaviour shifts occurring within instructional activity intervals.
   * `export_teaching_activity_results`: Multi-CSV serialization.
3. **Visual Analytics Engine (`src/activity/visualization.py`):**
   * `create_activity_timeline_figure`: Horizontal Gantt-style timeline chart with activity segment blocks, `MM:SS` timecodes, and dominant behaviour badges.
   * `create_activity_behaviour_distribution_figure`: Grouped bar chart comparing observable behaviour percentage shares across teaching activities.
   * `create_activity_behaviour_heatmap_figure`: Matrix heatmap of [Behaviours × Activities] with percentage share annotations.
   * `create_track_activity_figure`: Track-specific activity breakdown chart.

### Decoupled Storage & CSV Export (`results/teaching_activity/<video_id>/`)
* **`teaching_activity_segments.csv`:** Validated instructional segments with start/end timecodes and annotation source.
* **`activity_behaviour_summary_<model>.csv`:** Activity-level summary table with dominant behaviours and tie indicators.
* **`activity_behaviour_distributions_<model>.csv`:** Activity-behaviour cross-distribution matrix with observed seconds and percentage shares.
* **`activity_transitions_<model>.csv`:** Activity-stratified transition frequencies.

* **`activity_transitions_<model>.csv`:** Activity-stratified transition frequencies.

---

## 16. Feature 11 — Research Experiments, Ablation & Temporal Error Analysis

Feature 11 provides rigorous academic evaluation of the classroom video temporal modeling pipeline through controlled comparative benchmarks, single-variable ablation studies, and in-depth temporal error analysis.

### Core Scientific Components

1. **4-Way Baseline Model Comparison:**
   * **Frame-Level CNN Baseline (`FrameCNNClassifier`):** Non-recurrent static baseline mapping 512-dim CNN visual features ($x_{t_L}$) to behaviour logits via an MLP head ($512 \to 128 \to \text{ReLU} \to \text{Dropout} \to 6$), sharing identical training parameters, loss weighting, and track-grouped test split.
   * **Recurrent Architectures:** Benchmarked against CNN+RNN, CNN+LSTM, and CNN+GRU.
   * **Measured Values:** Evaluates actual Accuracy, Macro/Weighted Precision, Recall, Macro/Weighted F1-score, parameter counts, and inference latency (ms/sample). Strictly no invented numbers.

2. **Controlled Ablation Studies:**
   * **Architectural Recurrence Benefit:** Isolates empirical gain of temporal recurrence over static frame-level modeling ($\Delta F_1$).
   * **Sequence Window Length ($L$):** Compares $L \in \{5, 10, 15\}$ frames (~0.8s, ~1.7s, ~2.5s) to assess stability vs boundary lag.
   * **Temporal Window Stride ($S$):** Compares $S \in \{1, 2, 5\}$ frames to evaluate window density and resolution vs computational cost.
   * **Loss Class-Weighting Strategy:** Evaluates inverse-frequency weighted cross-entropy vs unweighted cross-entropy on minority behaviour recall.

3. **Temporal Error Analysis Across Time:**
   * **Transition Boundary Dynamics:** Quantifies error clustering near behaviour transitions ($|\Delta t| \le 0.5\text{s}$) vs steady-state intervals, calculating boundary error multipliers.
   * **Duration Tier Sensitivity:** Stratifies error rates into Fleeting (< 2.0s), Moderate (2.0s–5.0s), and Sustained ($\ge 5.0\text{s}$) episodes.
   * **Visually Similar Behaviour Pairs:** Diagnoses confusion between ambiguous actions (*Looking toward instruction* $\leftrightarrow$ *Looking away*, *Reading/writing* $\leftrightarrow$ *Mobile-device activity*).
   * **Activity-Conditioned Error Rates:** Assesses misclassification rates stratified across teaching activities (Lecture, Discussion, Problem-solving, Presentation).
   * **Concrete Misclassification Case Studies:** Extracts representative test errors with diagnostic contextual rationales.

4. **Per-Class Breakdown & Dataset Limitations:**
   * Full precision, recall, F1, and support across all 6 canonical classes.
   * Automated limitation alerts for under-represented ($N < 5$) or missing ($N = 0$) classes in the evaluated test partition.

5. **Evidence-Based Pedagogical Conclusions:**
   * Programmatic synthesis of peer-review-ready conclusion statements grounded strictly in actual measured numbers.
   * Enforces zero internal mental state inferences (strictly prohibits terms like "attention", "boredom", "comprehension", "motivation").

### Feature 11 File Outputs

* **`baseline_comparison.csv`:** Tabular comparison of Frame-level CNN, RNN, LSTM, and GRU test metrics.
* **`ablation_summary.csv`:** Controlled trial records with baseline vs ablated condition metrics and $\Delta F_1$.
* **`temporal_error_records.csv`:** Sequence-level error logs with boundary distance, duration tier, and diagnosis notes.
* **`per_class_metrics.csv`:** Granular 6-class metrics table with representation status.

---

## 17. Automated Testing

Run the complete PyTest suite covering video validation, preprocessing, frame sampling, person detection, multi-object tracking, observable behaviour recognition, CNN visual feature extraction, temporal sequence creation, recurrent sequence modelling, trajectory extraction, teaching activity analysis, research experiments & ablation, and Streamlit UI workflows:

```bash
pytest tests/ -v
```

The **166-test automated suite** (100% pass rate) covers:
* `test_video_utils.py` (14 tests): Filename sanitization, path traversal prevention, extension validation, OpenCV decodability, empty/corrupt file rejection, metadata extraction.
* `test_frame_extractor.py` (13 tests): Image validation, color conversion, resizing, chronological timestamp ordering, sampling ratios, CSV schema verification, cache handling.
* `test_detector.py` (7 tests): YOLO model initialization, person detection inference on classroom scenes, confidence threshold filtering, bounding box rendering, empty/zero-person frame handling, invalid inputs, and batch pipeline execution.
* `test_tracker.py` (9 tests): Tracker initialization (ByteTrack & BoT-SORT), persistent color generation, consecutive frame tracking continuity, confidence threshold filtering, zero-person handling, trajectory rendering, tracker reset, and end-to-end `tracks.csv` schema validation.
* `test_behaviour.py` (12 tests): Target behaviour labels & metadata, person crop preprocessing & clipping, invalid crop rejection, prototype mode initialization, visual heuristic prediction, peer proximity logic, blur/unknown handling, visual badge drawing, mock PyTorch model forward pass, and end-to-end pipeline execution with `behaviours.csv` validation.
* `test_cnn.py` (11 tests): ResNet18 model loading, classification head removal, CPU/CUDA device auto-detection, single-crop extraction (512D), batch extraction (B, 512), invalid/empty/out-of-bounds crop safety, end-to-end pipeline execution on synthetic video sequences, temporal order preservation, behaviour label linking, and 2D PCA projection/figure generation.
* `test_temporal.py` (22 tests): Parameter validation, sliding-window count formula verification, chronological frame index sorting, short track skipping policy, gap splitting policy, multi-track isolation, dominant behaviour calculation, PyTorch FloatTensor conversion, `ClassroomSequenceDataset` DataLoader batching, end-to-end pipeline execution with `.npy` + `.csv` file output, and timeline/coverage visualizations.
* `test_temporal_models.py` (20 tests): Target behaviour class integer encoding, track-grouped train/val/test splitting strictly preventing overlapping window leakage, training set class weighting, PyTorch Dataset & DataLoader factories, forward pass & output shape verification for RNN, LSTM, and GRU, parameter count verification, training loop execution, fair comparison multi-model training, early stopping validation, test set evaluation metrics calculation, comparison table generation, single-sequence inference, batch prediction CSV generation, training curves plotting, confusion matrix plotting, and model comparison plotting.
* `test_trajectory.py` (16 tests): Timestamp formatting (`MM:SS`), track trajectory isolation, time-range window filtering, tracking gap detection, consecutive segment merging, chronological transition calculation, duration and distribution calculation, trajectory summary generation, unknown/uncertain label preservation, zero-retraining inference reuse with existing checkpoints, CSV export schema verification, categorical timeline figure generation, duration distribution figure generation, transitions figure generation, multi-model comparison timeline generation, and classroom multi-track overview figure generation.
* `test_activity.py` (16 tests): Activity labels & canonical constants, segment dataclass properties, temporal validation (negative start, start >= end, exceeds duration, invalid activity, overlap detection), segment persistence and loading roundtrip, default demo segment generator, chronological prediction mapping, observed duration & percentage share calculation, summary table generation & explicit tie detection, activity-stratified transitions, CSV export verification, and visual analytics figures (timeline, grouped bars, heatmap, track breakdown).
* `test_experiments.py` (12 tests): Frame-level CNN forward pass shapes (2D & 3D), baseline training loop & checkpoint creation, 4-way baseline comparison engine, architecture ablation trial generation, loss weighting ablation execution, transition boundary distance calculation, duration tier categorization, visual similarity matching, temporal error analysis engine & case studies, per-class metrics reporting & representation status detection, top confused pairs extraction, evidence-based conclusions validation with zero mental-state terms, and all 6 visualization figure routines.
* `test_app.py` (14 tests): Streamlit end-to-end UI integration tests covering initial render, file upload, metric cards, extraction button triggers, detection workflows, Feature 4 tracking workflows, Feature 5 behaviour recognition workflows, Feature 6 CNN extraction workflows, Feature 7 temporal sequence creation workflows, Feature 8 recurrent temporal modelling UI workflows, Feature 9 observable behaviour trajectory workflows, Feature 10 teaching activity analysis workflows, Feature 11 research experiments workflows, and corrupted upload handling.

---

## 18. Current Limitations (Features 1–11 Scope)

Features 1 through 11 focus on **Classroom Video Ingestion, Preprocessing, Frame Extraction, Person Detection, Multi-Object Tracking, Observable Behaviour Recognition, CNN Visual Feature Extraction, Temporal Sequence Creation, Recurrent Temporal Sequence Modelling, Observable Behaviour Trajectory Profiling, Teaching Activity Analysis, and Research Experiments with Baseline Comparison, Ablation Studies, and Temporal Error Dynamics**.

Current limitations:
* Teaching activity labels are based on designated interval segments (manual or curriculum metadata); automated multimodal video-audio activity detection is out of scope.
* Models represent observable physical classroom behaviours and do not infer mental engagement, cognitive focus, comprehension, or motivation.
* Dataset sample distribution: Evaluated single-clip partitions may possess natural class imbalances, with certain behaviour classes having low or zero test representation; multi-classroom cross-validation is recommended for broader generalization.

---

## 19. Future Research Pipeline Roadmap

The subsequent development phases will follow this structured academic pipeline:

```text
Classroom Video Input (Feature 1 — Completed)
   ↓
Frame Extraction & Preprocessing (Feature 2 — Completed)
   ↓
Student Detection (YOLO / Spatial Bounding Boxes) (Feature 3 — Completed)
   ↓
Student Tracking (ByteTrack / BoT-SORT Multi-Object Tracking) (Feature 4 — Completed)
   ↓
Observable Behaviour Recognition (Spatial Action Analysis) (Feature 5 — Completed)
   ↓
CNN Visual Feature Extraction (Spatial Representations) (Feature 6 — Completed)
   ↓
Temporal Sequence Creation (Sliding Window Time Sequences) (Feature 7 — Completed)
   ↓
Sequence Modeling (RNN / LSTM / GRU) (Feature 8 — Completed)
   ↓
Observable Behaviour Trajectory Profiling (Feature 9 — Completed)
   ↓
Teaching Activity Analysis (Feature 10 — Completed)
   ↓
Research Experiments, Ablation & Temporal Error Analysis (Feature 11 — Completed)
   ↓
Final Interactive Analytics Dashboard (Feature 12)
```


