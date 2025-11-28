Water Level Gauge Detection and Reading
(YOLOv8 Segmentation + YOLOv8 Detection + OCR)

This project is a complete computer-vision pipeline to automatically detect river water level gauges from field images, segment the gauge region, detect printed gauge numerals, and estimate the actual water level. The system uses two YOLOv8 models plus OCR:

Segmentation Model – Detects and masks the gauge scale.

Number Detection Model – Detects printed gauge numbers (150, 200, 250, 300, 350, etc).

OCR (EasyOCR) – Optional digit reading & sanity checks.

Additional utilities handle COCO-to-YOLO polygon conversion, dataset structuring, training, inference, and model export to TFLite.

📌 Goals of the System

Detect the physical gauge ruler in field images using a segmentation model.

Crop the gauge region and run a number detection model on the ROI.

Optionally validate detected classes using OCR.

Convert COCO segmentation datasets into YOLOv8 polygon format.

Train both models in Google Colab or locally.

Export both models to TFLite for Android deployment.

Estimate the final water level in centimeters using calibration logic.

🔄 Overall Pipeline (High-Level)
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

🧠 Model Overview
1️⃣ Segmentation Model (YOLOv8-Seg)

Class: water_level_gauge

Produces:
✔ segmentation mask
✔ bounding box

Purpose: isolate gauge so downstream logic ignores background clutter.

2️⃣ Number Detection Model (YOLOv8)

Classes: 150, 200, 250, 300, 350 (or your custom set)

Detect discrete printed numbers on the gauge.

Each detected class is mapped to a real-world height via calibration.

🚀 Usage
✔ Training (CLI Recommended)
Segmentation Model
yolo task=segment mode=train model=yolov8n-seg.pt data=data_seg.yaml epochs=40 imgsz=640

Number Detection Model
yolo task=detect mode=train model=yolov8n.pt data=data_num.yaml epochs=40 imgsz=640

✔ Inference (Python API)
from ultralytics import YOLO
import easyocr

seg_model = YOLO("best_seg.pt")
num_model = YOLO("best_num.pt")
ocr = easyocr.Reader(['en'])

# 1) run segmentation → get gauge mask & bbox
# 2) crop gauge region
# 3) run number detection model on crop
# 4) optional OCR and water-level computation


The script also supports overlay visualization of masks, bounding boxes, and predicted levels.

📁 Dataset Structure
dataset_root/
├── seg/
│   ├── train/images/
│   ├── train/labels/
│   ├── valid/images/
│   ├── valid/labels/
│   └── data_seg.yaml
└── num/
    ├── train/images/
    ├── train/labels/
    ├── valid/images/
    ├── valid/labels/
    └── data_num.yaml

🔧 Segmentation Conversion Pipeline
COCO Format (_annotations.coco.json)
        │
        ▼
Normalized YOLO Polygon Labels (*.txt)
        │
        ▼
YOLOv8 Segmentation Dataset

🔢 Number Detection Labels

Standard YOLO bounding box labels.

Example class mapping:

0 → 150
1 → 200
2 → 250
3 → 300
4 → 350

🏗 Model Architectures
Segmentation
Input → YOLOv8 Backbone
         │
         ├── Segmentation Head → gauge mask
         └── Detection Head → gauge box

Number Detection
Input (cropped ROI) → YOLOv8 Backbone → Detection Head

📤 Export to TFLite
from ultralytics import YOLO

YOLO("best_seg.pt").export(format="tflite", imgsz=640, nms=False)
YOLO("best_num.pt").export(format="tflite", imgsz=640, nms=True)


Exported .tflite models are integrated into your Android app.

📊 Results
Segmentation Model

High mAP

Clean masks even in noisy real-world scenes

Number Detection Model

Very high accuracy on curated dataset

Robust detection of classes

End-to-End Pipeline

Stable water level estimates

Reliable even with reflections, lighting changes, clutter

Artifacts such as PR curves, training graphs, and confusion matrices are stored under:

runs/segment/...
runs/detect/...

🤝 Contributing

Fork the repo and create pull requests.

Keep paths and configs flexible.

Maintain modular folder structure:

dataset_tools/  
training/  
inference/

📜 License

MIT License — free to use, modify, and distribute.

🙏 Acknowledgments

Ultralytics YOLOv8

EasyOCR

PyTorch

COCO Format

Google Colab + Drive
