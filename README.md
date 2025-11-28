Water Level Gauge Detection and Reading
(YOLOv8 Segmentation + YOLOv8 Detection + OCR)

This project is a complete computer‑vision pipeline to automatically detect river water level gauges in field images, segment the gauge region, read gauge numerals, and estimate the water level. The system uses two YOLOv8 models plus OCR:

A segmentation model to detect and mask the gauge scale.

A number detection model to detect fixed gauge numerals (e.g., 150, 200, 250, 300, 350).

EasyOCR (or a similar OCR utility) for optional digit reading and sanity checks.

Additional utilities handle COCO‑to‑YOLO polygon conversion, dataset organization, and end‑to‑end training/inference in Google Colab or locally.

Description
The goals of this system are to:
Detect the gauge ruler in field photos and produce accurate segmentation masks and bounding boxes.
Detect printed gauge numbers using a separate YOLOv8 detection model.
Optionally read numerals with OCR and cross‑check with detected classes.
Convert COCO‑style segmentation datasets into YOLO polygon labels for training YOLOv8‑seg.
Run fully in Google Colab + Google Drive (GPU training) with equivalent local / CLI workflows.
Export both models to TFLite for mobile deployment.

Overall Pipeline
High‑Level Workflow
text
┌─────────────────┐
│  Input Images   │
│  (Field Photos) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ YOLOv8 Segmentation     │
│ (Gauge Seg Model)       │
└────────┬────────────────┘
         │  gauge mask + box
         ▼
┌─────────────────────────┐
│  Crop Gauge Region      │
└────────┬────────────────┘
         │  cropped ROI
         ▼
┌─────────────────────────┐
│ YOLOv8 Detection        │
│ (Gauge Number Model)    │
└────────┬────────────────┘
         │  class IDs: 150/200/…
         ▼
┌─────────────────────────┐
│ Optional EasyOCR Check  │
│ (Digits / Consistency)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Water Level Estimator  │
│  (Calibrated Output)    │
└─────────────────────────┘
Model 1 – Segmentation: isolates the physical gauge so downstream logic can safely ignore background, reflections, etc.

Model 2 – Detection: detects the discrete gauge numbers on the ruler. Each detection is mapped to its real‑world height using a calibration table.

Water‑level logic: interpolates between nearest detected numbers and the segmented waterline to give a final water level in centimeters.

Usage
Training and inference are implemented using both the Ultralytics CLI and the Python API.

CLI (recommended)
Prepare YOLO data configs:

data_seg.yaml – for the segmentation model with class: water_level_gauge.
data_num.yaml – for the number detection model with classes: 150, 200, 250, 300, 350 (or your exact set).

Train segmentation model:

bash
yolo task=segment mode=train model=yolov8n-seg.pt data=data_seg.yaml epochs=40 imgsz=640
Train number detection model:

bash
yolo task=detect mode=train model=yolov8n.pt data=data_num.yaml epochs=40 imgsz=640
Run inference on a folder of images (segmentation then detection on cropped ROI).

Python API
A unified script (e.g., river2_py.py) contains:

Google Drive mounting (for Colab).

COCO→YOLO conversion helpers.

Training calls for both models.

Inference pipeline:

python
from ultralytics import YOLO
import easyocr

seg_model = YOLO("best_seg.pt")
num_model = YOLO("best_num.pt")
ocr = easyocr.Reader(['en'])

# 1) run segmentation → get gauge mask & bbox
# 2) crop gauge region
# 3) run number detection model on crop
# 4) optional OCR and water-level computation
The same file also includes visualization utilities to overlay masks, boxes, and predicted water levels.

Dataset
Structure
text
dataset_root/
├── seg/               # segmentation dataset
│   ├── train/images/
│   ├── train/labels/
│   ├── valid/images/
│   ├── valid/labels/
│   └── data_seg.yaml
└── num/               # number detection dataset
    ├── train/images/
    ├── train/labels/
    ├── valid/images/
    ├── valid/labels/
    └── data_num.yaml
Segmentation Conversion Pipeline
text
COCO Format (_annotations.coco.json)
        │
        ▼
Normalized YOLO Polygon Labels (*.txt)
        │
        ▼
YOLOv8 Segmentation Dataset
Source: COCO instance‑segmentation JSON per split (train/valid/test).

Conversion: Custom script converts COCO polygons to YOLOv8 segmentation format for water_level_gauge.

Organization: Images moved under images/, labels under labels/ for each split.

Number Detection Labels
Classic YOLO box labels.

Each image has bounding boxes plus class IDs mapped to discrete gauge numbers:

0 → 150, 1 → 200, 2 → 250, 3 → 300, 4 → 350 (or your chosen mapping).

Models
Segmentation Model
text
Input (640×640)
  │
  ▼
YOLOv8 Backbone → Segmentation Head
  │                 ├─ gauge mask
  └ Detection Head ─┴─ gauge box
Task: segment the gauge ruler (single class: water_level_gauge).

Training: epochs ≈ 30–50, imgsz=640, batch=8–16, early stopping enabled.

Metrics: mAP@0.5 and mAP@0.5:0.95 for both box and mask; precision/recall and confusion matrix for the gauge class.

Number Detection Model
text
Input (cropped gauge ROI)
  │
  ▼
YOLOv8 Backbone → Detection Head
Task: detect printed numerals (150, 200, 250, …).

Outputs: bounding boxes and class IDs, one per fixed gauge number.

Use: detections are converted to real‑world heights via a calibration table for water‑level estimation.

Export for Deployment
Both models can be exported to TFLite for on‑device inference:

python
from ultralytics import YOLO

YOLO("best_seg.pt").export(format="tflite", imgsz=640, nms=False)
YOLO("best_num.pt").export(format="tflite", imgsz=640, nms=True)
These TFLite models are then integrated into the Android app and validated using a small TFLite inspection script (input/output tensor checks).

Results
Segmentation model:
High mAP and visually clean masks on validation images; reliably isolates the gauge even in noisy field scenes.

Number detection model:
mAP@0.5 ≈ very high (near‑perfect on the curated dataset), correctly detecting target numbers.

End‑to‑end pipeline:
Combined segmentation + detection + interpolation gives stable water‑level estimates in centimeters.
Training artifacts (curves, PR plots, confusion matrices, best weights) are stored under runs/segment/... and runs/detect/....

Contributing
Fork the repo and create a feature branch.
Keep dataset paths and model paths configurable (CLI args or config files).
Prefer modular Python scripts over large ad‑hoc notebooks:
dataset_tools/ – COCO→YOLO converters, visualization.
training/ – scripts for seg and number models.
inference/ – end‑to‑end water‑level demo.

When adding features:
Provide example CLI commands.
Test both models and the combined pipeline.
For bugs or feature ideas, open an Issue.

License
MIT License – you may use, modify, and distribute this project. If you change the license, update the LICENSE file and this section accordingly.

Acknowledgments
Ultralytics YOLOv8 – backbone for segmentation and detection.​
EasyOCR – OCR engine for optional numeral reading and validation.​​
PyTorch ecosystem – training and deep learning infrastructure.​
COCO format – base annotation format for the original segmentation dataset.​
Google Colab & Drive – primary development and training environment.
