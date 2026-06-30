
# Computer Vision: Object Detection and Re-Identification (ReID)

## Project Overview

This project implements a complete Computer Vision pipeline for **Object Detection, Localization, Tracking, and Re-Identification (ReID)** using **YOLO11**, **OpenCV**, **PyTorch**, and **BoT-SORT**.

The system is capable of detecting multiple objects in images and videos, localizing them using bounding boxes, assigning persistent tracking IDs across video frames, and performing inference on recorded or live video streams.

The YOLO11 model was fine-tuned on the **COCO128** dataset and evaluated using standard object detection metrics such as **Precision**, **Recall**, and **mAP**.

---

## Features

- Object Detection using YOLO11
- Bounding Box Localization
- Multi-Object Tracking using BoT-SORT
- Re-Identification (Persistent Object IDs)
- Image Inference
- Video Inference
- Fine-tuned YOLO11 Model
- Model Evaluation
- OpenCV Integration
- PyTorch-based Deep Learning Pipeline

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Ultralytics YOLO11 | Object Detection |
| OpenCV | Image & Video Processing |
| BoT-SORT | Multi Object Tracking & ReID |

---

## Project Structure

```
Object_Detection_And_ReID/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── models/
│
├── outputs/
│
├── runs/
│
├── src/
│   ├── detection/
│   ├── training/
│   ├── tracking/
│   └── evaluation/
│
├── requirements.txt
├── main.py
└── README.md
```

---

## Dataset

The project uses the **COCO128** dataset for training and evaluation.

Dataset contains:

- 128 Images
- Object Bounding Boxes
- Class Labels
- YOLO Annotation Format

---

## Workflow

```
Input Image / Video
          │
          ▼
      OpenCV
(Read Image / Frames)
          │
          ▼
   Trained YOLO11 Model
          │
          ▼
 Object Detection
(Class + Confidence + Bounding Box)
          │
          ▼
    BoT-SORT Tracker
          │
          ▼
Persistent Object IDs (ReID)
          │
          ▼
 Display Annotated Output
```

---

## Training

The pretrained **YOLO11n** model was fine-tuned on the COCO128 dataset.

Training included:

- Transfer Learning
- Data Augmentation
- Automatic Validation
- Model Checkpoint Saving

Generated weights:

```
best.pt
last.pt
```

---

## Built-in Data Augmentation

The training pipeline automatically applied:

- Mosaic Augmentation
- Horizontal Flip
- HSV Color Augmentation
- Random Scaling
- Translation
- Random Erasing

---

## Evaluation Metrics

Model Performance:

| Metric | Value |
|---------|-------|
| Precision | **80.38%** |
| Recall | **67.25%** |
| mAP@0.50 | **76.20%** |
| mAP@0.50:0.95 | **58.26%** |

---

## Inference

The trained model performs inference on:

- Images
- Recorded Videos
- Live Webcam Feed

Output includes:

- Object Class
- Confidence Score
- Bounding Box
- Tracking ID

---

## Tracking & Re-Identification

Object tracking is implemented using **BoT-SORT**, which maintains persistent IDs across consecutive video frames.

Example:

```
Frame 1

Person → ID 1

Person → ID 2

↓

Frame 2

Person → ID 1

Person → ID 2
```

This allows the system to continuously identify previously detected objects.

---

## Results

The project successfully performs:

- Object Detection
- Localization
- Multi-Object Tracking
- Re-Identification
- Video Inference
- Model Evaluation

---

## Future Improvements

- Train on larger custom datasets
- Real-time GPU deployment
- Vehicle and Traffic Analytics
- Person Counting
- Intrusion Detection
- Face Recognition Integration
- Edge Device Deployment
- Model Quantization for Faster Inference

---

## References

- Ultralytics YOLO Documentation
- PyTorch Documentation
- OpenCV Documentation
- BoT-SORT GitHub Repository

---

## Author

**Vishal Badhan**

Computer Vision | Machine Learning | Deep Learning