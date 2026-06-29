from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="datasets/raw/coco128/coco128.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    project="runs/train",
    name="submission"
)