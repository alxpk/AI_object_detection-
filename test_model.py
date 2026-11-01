
from ultralytics import YOLO

# Load the trained model from its actual location
model = YOLO(
    "runs/detect/runs/detect/coco8_experiment/weights/best.pt"
)

print("Custom YOLO model loaded successfully!")

# Test using your webcam
model.predict(
    source=0,
    show=True,
    conf=0.25,
    device="cpu"
)
