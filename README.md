# AI Object Detection System 🎯

A real-time AI object detection application built with **YOLO** and **OpenCV**, designed for live webcam inference, custom model training, and performance monitoring.

> **Status:** 🚧 Work In Progress (WIP) — early development stage.

---

## 🚀 Features

- **Live Webcam Inference (`main.py`)**: Real-time object detection feed with bounding box visualization and confidence scores.
- **HUD & Performance Stats**: Real-time calculation and display of **FPS** and **total object counts** directly on the video stream.
- **Custom Model Training (`train.py`)**: Training pipeline using Ultralytics YOLO with configurable epochs, batch size, and dataset configs.
- **Model Evaluation (`test_model.py`)**: Quick webcam evaluation script to test custom-trained weights.

---

## 📁 Project Structure

```text
AI Detection Project/
├── main.py          # Real-time webcam object detection with HUD stats
├── train.py         # YOLO model training script (COCO8 dataset experiment)
├── test_model.py    # Evaluation script for custom trained weights
├── yolo26n.pt       # Pretrained YOLO model weights
├── requirements.txt # Python package dependencies
├── .gitignore       # Git ignore rules for virtualenvs, cache, and runs
└── README.md        # Project documentation
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+ installed
- Webcam connected

### 2. Clone the Repository
```bash
git clone https://github.com/alxpk/AI_object_detection-.git
cd AI_object_detection-
```

### 3. Setup Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Real-Time Detection
Run the main script to start live webcam detection:
```bash
python main.py
```
- Press **Q** to exit the detection window.

### Train Model
Run the training script to train YOLO on the COCO8 dataset:
```bash
python train.py
```

### Test Trained Weights
Evaluate your trained checkpoint:
```bash
python test_model.py
```

---

## 🗺️ Roadmap & Upcoming Features

- [ ] Custom class filtering (detect specific objects only)
- [ ] Add sound / webhook alert triggers on detection
- [ ] Graphical User Interface (GUI / Web Dashboard)
- [ ] Export support (ONNX / TensorRT for edge deployment)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
