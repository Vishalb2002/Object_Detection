# from src.detection.model import YOLOModel
# from src.detection.image_detector import ImageDetector


# def main():

#     # Load YOLO
#     detector = YOLOModel()
#     model = detector.get_model()

#     # Image Detector
#     image_detector = ImageDetector(model)

#     results = image_detector.detect("datasets/test/000000000510.jpg")

#     result = results[0]

#     print(result.boxes)


# if __name__ == "__main__":
#     main()

from src.detection.model import YOLOModel
from src.detection.video_detector import VideoDetector


def main():

    detector = YOLOModel(
        "runs/detect/runs/train/submission-2/weights/best.pt"
    )
    model = detector.get_model()

    video = VideoDetector(model)

    video.detect("datasets/test/demo.mp4")


if __name__ == "__main__":
    main()