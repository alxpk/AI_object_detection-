
from ultralytics import YOLO

def main():
    # Load the same pretrained model as our webcam
    model = YOLO("yolo26n.pt")

    # Train on the COCO8 dataset
    results = model.train(
        data="coco8.yaml",
        epochs=3,
        imgsz=320,
        batch=2,
        device="cpu",
        workers=0,
        project="runs/detect",
        name="coco8_experiment"
    )

    print("Training completed!")

if __name__ == "__main__":
    main()
