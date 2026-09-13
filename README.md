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

## 2. Technology Stack (Features 1, 2, 3, 4, 5 & 6)

* **Programming Language:** Python 3.12+ (supports Python 3.11+)
* **Web Application Framework:** Streamlit
* **Computer Vision & Video Processing:** OpenCV (`opencv-python`)
* **Object Detection & Deep Learning:** Ultralytics YOLO (`ultralytics`), PyTorch (`torch`, `torchvision`)
* **Multi-Object Tracking:** ByteTrack & BoT-SORT (Linear Assignment Problem solver `lap`)
* **Pretrained CNN Visual Backbone:** PyTorch Torchvision ResNet18 (512-dim visual embeddings)
* **Visualization & Plotting:** Matplotlib (`matplotlib`), Pillow (`Pillow`)
* **Numerical Computing & SVD/PCA:** NumPy
* **Data Structures & Processing:** Pandas
* **Test Suite:** PyTest

*(Future sequence modeling modules such as RNN/LSTM/GRU will be introduced in Feature 7 & 8).*

---

## 3. Project Structure

```text
EduPulse_AI/
│
├── app.py                              # Streamlit main application entry point (Features 1-6)
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
│           └── cnn_features_metadata.csv # Feature 6 spatial-temporal metadata mapping
│
├── models/                             # Model weights directory
│   └── yolov8n.pt                      # Pretrained YOLOv8n detector (~6.2 MB)
│
├── notebooks/                          # Research & exploratory notebooks
│
├── src/                                # Modular source code
│   ├── __init__.py
│   ├── config.py                       # Central paths, sampling defaults, CNN & tracking configs
│   ├── video/                          # Video ingestion & frame extraction (Feature 1 & 2)
│   │   ├── __init__.py
│   │   ├── video_utils.py              # Video validation, metadata extraction, sanitization
│   │   └── frame_extractor.py          # Chronological extraction, sampling, & metadata engine
│   ├── preprocessing/                  # Image validation, color handling, & resizing
│   │   ├── __init__.py
│   │   └── frame_preprocessor.py       # FramePreprocessor utility class
│   ├── detection/                      # Student / Person Detection (Feature 3)
│   │   ├── __init__.py
│   │   └── detector.py                 # YOLOPersonDetector & detection pipeline
│   ├── tracking/                       # Student / Person Tracking (Feature 4)
│   │   ├── __init__.py
│   │   └── tracker.py                  # PersonTracker (ByteTrack/BoT-SORT), TrackResult, & trajectory engine
│   ├── behaviour/                      # Observable behaviour classification (Feature 5)
│   │   ├── __init__.py
│   │   ├── behaviour_classifier.py     # BehaviourClassifier & batch recognition pipeline
│   │   ├── behaviour_labels.py         # Canonical class constants, palette & descriptions
│   │   └── preprocessing.py            # Person crop extraction, clipping & normalization
│   ├── cnn/                            # CNN Visual Feature Extraction (Feature 6)
│   │   ├── __init__.py
│   │   ├── feature_extractor.py        # CNNFeatureExtractor (ResNet18 512D) & batch extraction
│   │   └── visualization.py            # NumPy SVD 2D PCA projection & scatter plotting
│   └── temporal/                       # Sequence modeling (RNN/LSTM/GRU) (Feature 7 & 8)
│
├── results/                            # Evaluation logs and ablation outputs (future)
├── tests/                              # Automated unit and integration tests (75 tests)
│   ├── __init__.py
│   ├── test_app.py                     # Streamlit UI integration tests (Features 1-6)
│   ├── test_video_utils.py             # Video validation unit tests
│   ├── test_frame_extractor.py         # Frame extraction, sampling, & preprocessor tests
│   ├── test_detector.py                # YOLO person detection & annotation unit tests
│   ├── test_tracker.py                 # Multi-object tracking, ID consistency, & trajectory tests
│   ├── test_behaviour.py               # Observable behaviour recognition unit tests
│   └── test_cnn.py                     # CNN ResNet18 loading, batch extraction, & PCA tests
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

## 12. Automated Testing

Run the complete PyTest suite covering video validation, preprocessing, frame sampling, person detection, multi-object tracking, observable behaviour recognition, CNN visual feature extraction, and Streamlit UI workflows:

```bash
pytest tests/ -v
```

The **75-test automated suite** covers:
* `test_video_utils.py` (14 tests): Filename sanitization, path traversal prevention, extension validation, OpenCV decodability, empty/corrupt file rejection, metadata extraction.
* `test_frame_extractor.py` (13 tests): Image validation, color conversion, resizing, chronological timestamp ordering, sampling ratios, CSV schema verification, cache handling.
* `test_detector.py` (7 tests): YOLO model initialization, person detection inference on classroom scenes, confidence threshold filtering, bounding box rendering, empty/zero-person frame handling, invalid inputs, and batch pipeline execution.
* `test_tracker.py` (9 tests): Tracker initialization (ByteTrack & BoT-SORT), persistent color generation, consecutive frame tracking continuity, confidence threshold filtering, zero-person handling, trajectory rendering, tracker reset, and end-to-end `tracks.csv` schema validation.
* `test_behaviour.py` (12 tests): Target behaviour labels & metadata, person crop preprocessing & clipping, invalid crop rejection, prototype mode initialization, visual heuristic prediction, peer proximity logic, blur/unknown handling, visual badge drawing, mock PyTorch model forward pass, and end-to-end pipeline execution with `behaviours.csv` validation.
* `test_cnn.py` (11 tests): ResNet18 model loading, classification head removal, CPU/CUDA device auto-detection, single-crop extraction (512D), batch extraction (B, 512), invalid/empty/out-of-bounds crop safety, end-to-end pipeline execution on synthetic video sequences, temporal order preservation, behaviour label linking, and 2D PCA projection/figure generation.
* `test_app.py` (9 tests): Streamlit end-to-end UI integration tests covering initial render, file upload, metric cards, extraction button triggers, detection workflows, Feature 4 tracking workflows, Feature 5 behaviour recognition workflows, Feature 6 CNN extraction workflows, and corrupted upload handling.

---

## 13. Current Limitations (Features 1–6 Scope)

Features 1 through 6 focus on **Classroom Video Ingestion, Preprocessing, Frame Extraction, Person Detection, Multi-Object Tracking, Observable Behaviour Recognition, and CNN Visual Feature Extraction**.

Current limitations:
* Pretrained ImageNet features capture general visual representations (posture, objects, appearance) but have not been fine-tuned on custom classroom datasets.
* Temporal sequence modeling (RNN / LSTM / GRU) is not yet implemented (scheduled for Features 7 & 8).
* Feature vectors represent visual spatial snapshots, not mental engagement or internal cognitive states.
* Feature sequences have not yet been framed into sliding temporal windows.

---

## 14. Future Research Pipeline Roadmap

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
Temporal Sequence Creation (Sliding Window Time Sequences) (Feature 7 — Upcoming)
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

