
from ultralytics import YOLO
import cv2
import time

# 1. Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# 2. Open the webcam
cap = cv2.VideoCapture(0)

# Set camera capture resolution (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

# Create a resizable display window
window_name = "AI Object Detection System"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# Detection settings
confidence = 0.25

# FPS calculation
previous_time = time.perf_counter()

print("AI Object Detection started!")
print("Press Q to quit.")

while True:
    # 3. Capture a webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # 4. Detect objects
    results = model.predict(
        frame,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # 5. Draw bounding boxes and labels
    output = result.plot()

    # 6. Count objects in the current frame
    object_counts = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        score = float(box.conf[0])

        object_counts[name] = object_counts.get(name, 0) + 1

        print(
            f"\rDetected: {name} | Confidence: {score:.2f}",
            end=" " * 10
        )

    # 7. Calculate FPS
    current_time = time.perf_counter()
    elapsed = current_time - previous_time
    fps = 1 / elapsed if elapsed > 0 else 0
    previous_time = current_time

    # 8. Display information on the video
    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Confidence threshold: {confidence:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    total_objects = sum(object_counts.values())

    cv2.putText(
        output,
        f"Objects detected: {total_objects}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # 9. Display webcam output
    cv2.imshow(window_name, output)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nDetection stopped.")
