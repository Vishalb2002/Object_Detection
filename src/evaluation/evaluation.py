from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/train/submission-2/weights/best.pt"
)

metrics = model.val(
    data="datasets/raw/coco128/coco128.yaml"
)

print(metrics.box.map)      # mAP50-95
print(metrics.box.map50)    # mAP50
print(metrics.box.mp)       # Precision
print(metrics.box.mr)       # Recall