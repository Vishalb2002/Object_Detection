from ultralytics import YOLO


class ImageDetector:
    def __init__(self, model):
        self.model = model

    def detect(self, image_path):
        results = self.model(image_path)
        return results