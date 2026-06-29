
from ultralytics import YOLO


class YOLOModel:
    def __init__(self, model_path="runs/detect/runs/train/submission-2/weights/best.pt"):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")

    def get_model(self):
        return self.model