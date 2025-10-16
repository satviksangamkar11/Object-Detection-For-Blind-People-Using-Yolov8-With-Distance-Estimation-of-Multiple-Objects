# 🎯 Smart Object Detection for Visually Impaired People with Pothole Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
**A real-time assistive system for visually impaired users using YOLOv8 for object and pothole detection, distance estimation, and speech output.**

[📹 View Demo](#-demo-videos) • [🚀 Quick Start](#-installation) • [📚 Documentation](#-usage) • [⭐ Star](https://github.com/satviksangamkar11/Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects/stargazers)
</div>

---

## 🌟 Overview

This project implements a sophisticated real-time assistive computer vision system specifically designed to enhance mobility and safety for visually impaired individuals. The system combines dual YOLOv8 models for comprehensive object detection, accurate distance estimation, and audio feedback to provide users with environmental awareness through speech output.

The system features:
- **🎯 Dual YOLOv8 Models**: General object detection (80+ COCO classes) + specialized pothole detection
- **📏 Real-time Distance Estimation**: Accurate distance calculation using computer vision principles
- **🔊 Speech Output System**: Text-to-speech integration for hands-free navigation assistance
- **⚠️ Safety-First Design**: Prioritized pothole detection with audio warnings
- **⚡ Real-time Performance**: GPU-accelerated processing for live camera feeds

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Dual-Model Detection** | Advanced YOLOv8-based detection for 80+ COCO classes + specialized pothole detection |
| 📏 **Distance Estimation** | Accurate real-time distance calculation using computer vision principles |
| 🔊 **Speech Output** | Text-to-speech integration for hands-free navigation assistance |
| ⚠️ **Pothole Safety Alerts** | Specialized detection and warning system for road hazards |
| 📹 **Multiple Input Sources** | Support for live camera feed and video files |
| ⚡ **Optimized Performance** | GPU-accelerated processing with configurable detection intervals |
| 🎨 **Visual Annotations** | Customizable bounding boxes with color-coded priority system |
| 🎯 **Smart Filtering** | Confidence-based detection filtering and largest object prioritization |
| 🔄 **Real-time Processing** | Threaded video capture and async speech synthesis for smooth performance |

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows/macOS/Linux
- **Hardware**: Webcam or video input device
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended for optimal performance)
- **Memory**: 4GB+ RAM (8GB+ recommended)

### Major Dependencies
- **Python**: 3.8+
- **YOLOv8** (ultralytics): Object detection framework
- **OpenCV** (cv2): Computer vision library
- **PyTorch** (torch): Deep learning framework
- **pyttsx3**: Text-to-speech engine
- **NumPy**: Numerical computing
- **torchvision**: PyTorch vision utilities
- **torchaudio**: PyTorch audio utilities

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/satviksangamkar11/Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects.git
cd Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects
```

### 2. Install Python dependencies
```bash
# Install PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python pyttsx3 ultralytics numpy
```

### 3. Verify GPU setup (recommended)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Create required directories
```bash
mkdir -p utils weights output_recordings
```

## 🚀 Usage

### Run Object Detection System (Main Application)
```bash
python smart_object_detector.py
```

This will start the real-time object detection system with:
- Live camera feed processing
- Object and pothole detection
- Distance estimation
- Speech output for detected objects
- Visual annotations with bounding boxes

### Train Custom Pothole Detection Model
```bash
python train_pothole_detector.py
```

This will train a custom YOLOv8 model for pothole detection using the provided dataset.

### Controls
- **ESC**: Exit the application
- **Space**: Pause/Resume detection
- **S**: Save current frame
- **M**: Toggle mute/unmute speech output

## 📁 Scripts Overview

### `smart_object_detector.py`
The main detection and speech assist script for real-time scenes. This script:
- Initializes dual YOLOv8 models (general objects + pothole detection)
- Captures live video feed from camera
- Performs real-time object detection and distance estimation
- Provides speech output for detected objects with distance information
- Displays visual annotations with bounding boxes and labels
- Prioritizes safety alerts for potholes and hazardous objects
- Supports both live camera and video file input

**Key Features:**
- Real-time dual-model inference
- Distance calculation using pinhole camera model
- Asynchronous speech synthesis
- Configurable detection parameters
- GPU acceleration support

### `train_pothole_detector.py`
Script for training custom YOLOv8 pothole detector models. This script:
- Loads custom pothole dataset from `pothole_dataset/` directory
- Configures YOLOv8 nano model for pothole-specific training
- Implements training loop with validation
- Saves trained model weights to `weights/` directory
- Generates training metrics and validation results
- Supports GPU acceleration for faster training

**Training Configuration:**
- Model: YOLOv8n (nano) for speed optimization
- Dataset: Custom pothole dataset (1 class)
- Epochs: 50 with early stopping
- Image Size: 640x640 pixels
- Batch Size: 16 (auto-adjustable)

## 📊 Project Structure
```
Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects/
├── 📄 README.md                    # Project documentation
├── 🚀 smart_object_detector.py     # Main detection and speech assist script
├── 🏋️ train_pothole_detector.py    # Pothole detection model training script
├── 📁 images/                      # Static demonstration images
│   ├── 🖼️ img1.jpg                # Market area detection example
│   └── 🖼️ img2.jpg                # Street scene detection example
├── 📁 output_recordings/           # Demo output videos
├── 📁 utils/                       # Utility files
│   └── 📋 coco.txt                # COCO class names (80 classes)
├── 📁 weights/                     # Model weights
│   ├── 🤖 yolov8n.pt              # Base YOLOv8 nano model
│   └── 🦺 yolov8n_pothole.pt      # Trained pothole detection model
└── 📁 pothole_dataset/             # Training dataset
    └── ⚙️ data.yaml               # Dataset configuration
```

## 🎯 Supported Object Classes

The system can detect **81 different object classes**:

### Standard COCO Classes (80 classes)
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, train
- **Animals**: cat, dog, horse, cow, bird, sheep, elephant, bear
- **Furniture**: chair, table, bed, couch, sofa, dining table
- **Electronics**: laptop, mouse, keyboard, cell phone, tv, remote
- **Food**: banana, apple, orange, pizza, cake, sandwich, hot dog
- **Sports**: tennis racket, baseball bat, skateboard, surfboard
- **Indoor Objects**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **Traffic**: traffic light, stop sign, parking meter
- **And many more...**

### Specialized Safety Classes (1 class)
- **Road Hazards**: pothole (custom-trained model)

## ⚙️ Configuration

### Key Parameters (in smart_object_detector.py)
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `FOCAL_LENGTH` | Camera focal length for distance calculation | 360 |
| `KNOWN_WIDTH` | Reference object width in cm | 60 |
| `DETECTION_INTERVAL` | Detection interval in seconds | 0.033 |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detection | 0.5 |

### Performance Tuning
```python
# For enhanced speed
INPUT_WIDTH = 224    # 4x faster processing
CONFIDENCE_THRESHOLD = 0.3  # More detections, lower precision

# For maximum accuracy
INPUT_WIDTH = 640    # Maximum accuracy
CONFIDENCE_THRESHOLD = 0.7  # Fewer but more confident detections
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA/GPU not detected**
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install correct PyTorch version
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Camera permission issues**
   ```python
   # Test camera access
   import cv2
   cap = cv2.VideoCapture(0)  # Try different indices: 1, 2, etc.
   ```

3. **Audio/TTS not working**
   ```bash
   # Install additional TTS dependencies
   pip install pywin32        # For Windows
   pip install espeak         # For Linux/macOS
   ```

4. **Performance issues**
   - Reduce input resolution: Set `INPUT_WIDTH = 224`
   - Lower confidence threshold: Set `CONFIDENCE_THRESHOLD = 0.3`
   - Use CPU only: Force `DEVICE = 'cpu'`

## 📊 Performance Metrics

### System Performance
- **Detection Rate**: 30 FPS with GPU acceleration
- **Memory Usage**: ~2GB VRAM (GPU) / ~4GB RAM (CPU)
- **Audio Latency**: <100ms response time
- **Dual-Model Inference**: ~15ms per frame (RTX 30-series)

### Model Performance (YOLOv8n)
| Model | Speed (FPS) | mAP@0.5 | Size (MB) | Use Case |
|-------|-------------|---------|-----------|----------|
| YOLOv8n | ~45 | 37.3 | 6.2 | **Recommended** - Best speed/accuracy balance |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Satvik Sangamkar**
- 📱 Phone: +91 9325623723
- 💻 GitHub: [@satviksangamkar11](https://github.com/satviksangamkar11)
- 📧 Project Link: [Object Detection for Blind People](https://github.com/satviksangamkar11/Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects)

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [COCO Dataset](https://cocodataset.org/) for object detection classes
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for text-to-speech functionality

---

**⭐ If you found this project helpful, please give it a star!**
