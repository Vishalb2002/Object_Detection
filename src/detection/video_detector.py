import cv2

class VideoDetector:

    def __init__(self, model):
        self.model = model

    def detect(self, video_path):

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = self.model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                verbose=False
            )

            annotated_frame = results[0].plot()

            cv2.imshow("Object Detection + Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()