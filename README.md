# 🎯 Smart Object Detection for Visually Impaired People with Pothole Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced intelligent computer vision system designed to assist visually impaired individuals by providing real-time object detection with distance estimation and audio feedback using dual YOLOv8 models. The system includes specialized pothole detection for enhanced safety and navigation assistance.

## 🌟 Key Features

- **🔍 Dual-Model Detection**: Advanced YOLOv8-based detection for 80+ COCO classes + specialized pothole detection
- **📏 Distance Estimation**: Accurate distance calculation using computer vision principles
- **🔊 Audio Feedback**: Text-to-speech integration for hands-free navigation assistance
- **⚠️ Pothole Safety Alerts**: Specialized detection and warning system for road hazards
- **📹 Multiple Input Sources**: Support for live camera feed and video files
- **⚡ Optimized Performance**: GPU-accelerated processing with configurable detection intervals
- **🎨 Visual Annotations**: Customizable bounding boxes with color-coded priority system
- **🎯 Smart Filtering**: Confidence-based detection filtering and largest object prioritization
- **🔄 Real-time Processing**: Threaded video capture and async speech synthesis for smooth performance

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or video file
- Windows/macOS/Linux
- NVIDIA GPU with CUDA 11.8+ (recommended for optimal performance)
- 4GB+ RAM (8GB+ recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Object-Detection-Clean-Repo
   ```

2. **Install Python dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python pyttsx3 ultralytics numpy
   ```

3. **Verify GPU setup (recommended)**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 📁 Project Structure

```
Object-Detection-Clean-Repo/
├── README.md                    # Project documentation
├── smart_object_detector.py     # Main dual-model application (recommended)
├── train_pothole_detector.py    # Pothole detection model training script
├── utils/
│   └── coco.txt                # COCO class names (80 classes)
├── pothole_dataset/
│   └── data.yaml              # Dataset configuration for training
├── weights/                    # Model weights
│   ├── yolov8n.pt             # Base YOLOv8 nano model
│   └── yolov8n_pothole.pt     # Trained pothole detection model
├── output_recordings/          # Demo output videos (generated)
└── runs/                      # Training outputs (generated)
```

## 🎮 Usage

### 🚀 Main Application
```bash
# Run the main smart object detector with dual-model support
python smart_object_detector.py
```

### 📹 Video File Processing
```bash
# The script supports video file input (configure in the script)
python smart_object_detector.py
```

### 🏋️ Model Training (Pothole Detection)
```bash
# Train the pothole detection model
python train_pothole_detector.py
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default Value | Optimized For |
|-----------|-------------|---------------|---------------|
| `FOCAL_LENGTH` | Camera focal length for distance calculation | 360 | General use |
| `KNOWN_WIDTH` | Reference object width in cm | 60 | Average object size |
| `DETECTION_INTERVAL` | Detection interval in seconds | 0.033 | 30 FPS detection |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detection | 0.5 | Balanced accuracy/speed |

## 🔧 Technical Details

### Dual-Model Architecture

The system employs two specialized YOLOv8 models:

1. **Regular Object Detection Model** (`yolov8n.pt`)
   - Pre-trained on COCO dataset (80 classes)
   - General-purpose object recognition

2. **Pothole Detection Model** (`yolov8n_pothole.pt`)
   - Custom-trained on pothole dataset
   - Single-class specialization for road hazards

### Distance Estimation Algorithm

```
Distance (cm) = (Known Width × Focal Length) / Width in Pixels
```

## 🎯 Supported Object Classes

The system can detect **81 different object classes**:

### Standard COCO Classes (80 classes)
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, etc.
- **Animals**: cat, dog, horse, cow, bird, etc.
- **Furniture**: chair, table, bed, couch, etc.
- **Electronics**: laptop, mouse, keyboard, cell phone, tv, etc.
- **And many more**: Including traffic lights, stop signs, etc.

### Specialized Safety Classes (1 class)
- **Road Hazards**: **pothole** (custom-trained model)

## 📊 Performance

### System Performance (Optimized Configuration)

- **Input Resolution**: 320×320 (4× faster than 640×640)
- **Detection Rate**: 30 FPS with GPU acceleration
- **Memory Usage**: ~2GB VRAM (GPU) / ~4GB RAM (CPU)
- **Audio Latency**: <100ms response time
- **Dual-Model Inference**: ~15ms per frame (RTX 30-series)

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA/GPU not detected**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Model loading errors**
   - Ensure model files exist in `weights/` directory
   - YOLOv8n will auto-download on first run if missing

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [COCO Dataset](https://cocodataset.org/) for object detection classes
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for text-to-speech functionality

---

**⭐ If you found this project helpful, please give it a star!**
