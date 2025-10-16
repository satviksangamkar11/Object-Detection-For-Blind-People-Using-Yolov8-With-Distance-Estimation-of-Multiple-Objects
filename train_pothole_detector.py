"""
YOLOv8 Pothole Detection Model Training Script

This script trains a YOLOv8 model on the pothole dataset for integration
with the smart object detector system.
"""

import torch
from ultralytics import YOLO
import os


def check_gpu_availability():
    """Check if GPU is available and print device information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ Number of GPUs: {gpu_count}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return 'cuda'
    else:
        print("⚠ No GPU found. Training will use CPU (slower)")
        return 'cpu'


def train_pothole_detector():
    """Train YOLOv8 model on pothole dataset"""
    
    print("="*70)
    print("YOLOV8 POTHOLE DETECTION MODEL TRAINING")
    print("="*70)
    
    # Check GPU availability
    device = check_gpu_availability()
    print()
    
    # Configuration
    data_yaml = os.path.abspath("pothole_dataset/data.yaml")
    base_model = "weights/yolov8n.pt"  # Using existing YOLOv8n as base
    output_model_name = "yolov8n_pothole"
    
    print(f"📁 Dataset Config: {data_yaml}")
    print(f"🤖 Base Model: {base_model}")
    print(f"💾 Output Model: {output_model_name}")
    print()
    
    # Verify dataset exists
    if not os.path.exists(data_yaml):
        print(f"❌ Error: Dataset config not found at {data_yaml}")
        return
    
    if not os.path.exists(base_model):
        print(f"⚠ Base model not found at {base_model}")
        print("📥 Downloading YOLOv8n model...")
        base_model = "yolov8n.pt"  # Will auto-download
    
    print("🚀 Loading base model...")
    model = YOLO(base_model)
    
    print("\n" + "="*70)
    print("🎯 STARTING TRAINING")
    print("="*70)
    print("Training Parameters:")
    print("  • Epochs: 50")
    print("  • Image Size: 640x640")
    print("  • Batch Size: 16 (auto-adjust if needed)")
    print("  • Device:", device)
    print("  • Patience: 10 (early stopping)")
    print()
    
    try:
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=50,              # Number of training epochs
            imgsz=640,              # Image size
            batch=16,               # Batch size (will auto-adjust if GPU memory is low)
            device=device,          # Use GPU if available
            patience=10,            # Early stopping patience
            save=True,              # Save checkpoints
            project="runs/pothole_detect",  # Project folder
            name=output_model_name, # Run name
            exist_ok=True,          # Overwrite existing
            pretrained=True,        # Use pretrained weights
            optimizer='auto',       # Auto optimizer selection
            verbose=True,           # Verbose output
            seed=42,                # Random seed for reproducibility
            deterministic=True,     # Deterministic training
            single_cls=False,       # Not single class
            rect=False,             # Rectangular training
            cos_lr=False,           # Cosine learning rate scheduler
            close_mosaic=10,        # Disable mosaic augmentation in last N epochs
            resume=False,           # Resume training
            amp=True,               # Automatic Mixed Precision
            fraction=1.0,           # Train on fraction of dataset
            profile=False,          # Profile ONNX and TensorRT speeds
            freeze=None,            # Freeze first N layers
            # Data augmentation
            hsv_h=0.015,            # HSV-Hue augmentation
            hsv_s=0.7,              # HSV-Saturation augmentation
            hsv_v=0.4,              # HSV-Value augmentation
            degrees=0.0,            # Rotation augmentation
            translate=0.1,          # Translation augmentation
            scale=0.5,              # Scale augmentation
            shear=0.0,              # Shear augmentation
            perspective=0.0,        # Perspective augmentation
            flipud=0.0,             # Vertical flip augmentation
            fliplr=0.5,             # Horizontal flip augmentation
            mosaic=1.0,             # Mosaic augmentation
            mixup=0.0,              # Mixup augmentation
            copy_paste=0.0,         # Copy-paste augmentation
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Get the best model path
        best_model_path = f"runs/pothole_detect/{output_model_name}/weights/best.pt"
        last_model_path = f"runs/pothole_detect/{output_model_name}/weights/last.pt"
        
        print(f"\n📊 Training Results:")
        print(f"  • Best Model: {best_model_path}")
        print(f"  • Last Model: {last_model_path}")
        print(f"  • Training Logs: runs/pothole_detect/{output_model_name}/")
        
        # Validate the model
        print("\n" + "="*70)
        print("🔍 VALIDATING MODEL")
        print("="*70)
        
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=data_yaml)
        
        print(f"\n📈 Validation Metrics:")
        print(f"  • mAP50: {metrics.box.map50:.4f}")
        print(f"  • mAP50-95: {metrics.box.map:.4f}")
        print(f"  • Precision: {metrics.box.mp:.4f}")
        print(f"  • Recall: {metrics.box.mr:.4f}")
        
        # Copy best model to weights directory for easy access
        import shutil
        target_path = "weights/yolov8n_pothole.pt"
        os.makedirs("weights", exist_ok=True)
        shutil.copy(best_model_path, target_path)
        
        print(f"\n✅ Best model copied to: {target_path}")
        print(f"\n🎉 Model is ready for integration into smart_object_detector.py!")
        
        return best_model_path
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {str(e)}")
        print("\nTroubleshooting tips:")
        print("  1. Reduce batch size if out of memory")
        print("  2. Check dataset paths in data.yaml")
        print("  3. Ensure dataset has valid annotations")
        return None


if __name__ == "__main__":
    print("\n🔧 YOLOv8 Pothole Detection Training")
    print("="*70)
    
    # Check CUDA setup
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print()
    
    # Start training
    train_pothole_detector()
    
    print("\n" + "="*70)
    print("🏁 TRAINING PROCESS COMPLETED")
    print("="*70)

