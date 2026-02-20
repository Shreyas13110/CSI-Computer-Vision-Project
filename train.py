from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset1\data.yaml",
    epochs=8,
    imgsz=640
)
