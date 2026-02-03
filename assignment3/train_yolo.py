"""
YOLOv5s Training Script for Kinder Surprise Figures Detection

This script trains a YOLOv5s model with transfer learning on the custom dataset.
Uses pre-trained COCO weights for better feature extraction.
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# ============ Configuration ============
DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
PROJECT_DIR = Path(__file__).parent / "runs" / "yolo"
MODEL_NAME = "yolov5s_kinder"

# Training hyperparameters
IMG_SIZE = 640
BATCH_SIZE = 16  # Adjust based on GPU memory (8GB RTX 4060 can handle 16)
EPOCHS = 100
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ============ Training ============
def train():
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Data config: {DATA_YAML}")
    print(f"Output directory: {PROJECT_DIR / MODEL_NAME}")
    print("-" * 50)
    
    # Load YOLOv5s pre-trained on COCO
    # Transfer learning: backbone weights are preserved, detection head is adapted
    model = YOLO("yolov5su.pt")  # YOLOv5s ultralytics version with pre-trained weights
    
    # Train the model
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=MODEL_NAME,
        exist_ok=True,
        
        # Transfer learning settings
        pretrained=True,
        
        # Data augmentation (YOLOv5 defaults with some customization)
        hsv_h=0.015,        # HSV-Hue augmentation
        hsv_s=0.7,          # HSV-Saturation augmentation
        hsv_v=0.4,          # HSV-Value augmentation
        degrees=15.0,       # Rotation augmentation
        translate=0.1,      # Translation augmentation
        scale=0.5,          # Scale augmentation
        fliplr=0.5,         # Horizontal flip probability
        mosaic=1.0,         # Mosaic augmentation
        mixup=0.1,          # Mixup augmentation
        
        # Training settings
        patience=20,        # Early stopping patience
        save=True,          # Save checkpoints
        save_period=10,     # Save every N epochs
        cache=True,         # Cache images for faster training
        workers=8,          # Data loader workers
        
        # Validation
        val=True,           # Run validation during training
        plots=True,         # Generate training plots
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best model saved to: {PROJECT_DIR / MODEL_NAME / 'weights' / 'best.pt'}")
    print(f"Last model saved to: {PROJECT_DIR / MODEL_NAME / 'weights' / 'last.pt'}")
    
    return results


def validate(weights_path=None):
    """Run validation on the trained model."""
    if weights_path is None:
        weights_path = PROJECT_DIR / MODEL_NAME / "weights" / "best.pt"
    
    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Run validation
    results = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
    )
    
    print("\n" + "=" * 50)
    print("Validation Results")
    print("=" * 50)
    print(f"mAP@0.5:     {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision:   {results.box.mp:.4f}")
    print(f"Recall:      {results.box.mr:.4f}")
    
    return results


def predict(image_path, weights_path=None, conf_threshold=0.5):
    """Run inference on an image."""
    if weights_path is None:
        weights_path = PROJECT_DIR / MODEL_NAME / "weights" / "best.pt"
    
    model = YOLO(str(weights_path))
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=str(PROJECT_DIR),
        name="predictions",
        exist_ok=True,
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv5s Training for Object Detection")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "predict"],
                        help="Mode: train, val, or predict")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file")
    parser.add_argument("--source", type=str, default=None, help="Image path for prediction")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch
        train()
    elif args.mode == "val":
        validate(args.weights)
    elif args.mode == "predict":
        if args.source is None:
            print("Error: --source required for prediction mode")
        else:
            predict(args.source, args.weights)
