from ultralytics import YOLO

class YOLODetector:
    def __init__(self, conf=0.5):
        self.model = YOLO("models/helmetv2.pt")
        self.conf = conf
        self.names = self.model.names

    def get_results(self, frame):

        output = self.model(frame, verbose=False)
        results = output[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if confidence < self.conf:
                continue

            label = self.names[class_id]

            detections.append((
                int(x1), int(y1), int(x2), int(y2),
                label,
                confidence
            ))

        return detections



        