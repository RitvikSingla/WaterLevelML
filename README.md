# Water Level Gauge Detection and Reading  
### YOLOv8 Segmentation + YOLOv8 Detection + OCR

This project provides a complete computer-vision pipeline to automatically detect river water level gauges, segment them, detect printed gauge numerals, and estimate the water level.

The system uses:

- **YOLOv8 Segmentation** (gauge ruler isolation)  
- **YOLOv8 Detection** (gauge number detection)  
- **OCR (EasyOCR)** for optional digit verification  
- **COCO → YOLO polygon conversion utilities**  
- **TFLite export** for mobile deployment  

---

## Goals

- Detect and segment water level gauges in field images  
- Detect fixed printed gauge numbers (150, 200, 250, etc.)  
- Optionally cross-validate number detection using OCR  
- Convert COCO segmentation to YOLO polygon format  
- Train the models in Google Colab or locally  
- Export YOLO models to TFLite for Android app integration  
- Compute final water level using calibrated logic  
---

# Models

## 1. Segmentation Model (YOLOv8-Seg)

- Class: `water_level_gauge`
- Output: mask + bounding box  
- Purpose: isolate gauge ruler from background clutter

## 2. Number Detection Model (YOLOv8)

- Classes: 150, 200, 250, 300, 350  
- Output: bounding boxes + class IDs  
- Mapped to real-world heights for water-level estimation  

---

# Training

## Segmentation Model

yolo task=segment mode=train model=yolov8n-seg.pt data=data_seg.yaml epochs=40 imgsz=640
yolo task=detect mode=train model=yolov8n.pt data=data_num.yaml epochs=40 imgsz=640

Inference Pipeline (Python):
from ultralytics import YOLO
import easyocr

seg_model = YOLO("best_seg.pt")
num_model = YOLO("best_num.pt")
ocr = easyocr.Reader(['en'])

# 1) segmentation → gauge mask + box
# 2) crop gauge region
# 3) detection on cropped ROI
# 4) optional OCR + water level estimation

Dataset Structure:
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

COCO → YOLO Segmentation Conversion:
COCO JSON
   │
   ▼
YOLO Polygon Labels (*.txt)
   │
   ▼
YOLOv8 Segmentation Dataset

Number Detection Class Mapping:
0 → 150  
1 → 200  
2 → 250  
3 → 300  
4 → 350

Export to TFLite:
from ultralytics import YOLO

YOLO("best_seg.pt").export(format="tflite", imgsz=640, nms=False)
YOLO("best_num.pt").export(format="tflite", imgsz=640, nms=True)

Results
Segmentation Model

High mAP

Clean, robust masks even in noisy conditions

Number Detection Model

Very high accuracy for fixed printed numbers

End-to-End Performance

Stable and accurate water level estimation in centimeters

License:
MIT License

Acknowledgments:
Ultralytics YOLOv8
EasyOCR
PyTorch
COCO Dataset Format
Google Colab + Google Drive
