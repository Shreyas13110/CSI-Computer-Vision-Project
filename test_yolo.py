from ultralytics import YOLO

model = YOLO("helmetv2.pt")

metrics = model.val(
    data="dataset1\data.yaml",
    split="test"
)

print(metrics)