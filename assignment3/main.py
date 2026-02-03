#!/usr/bin/env python3
"""
Object Detection Training Pipeline
==================================

This is the main entry point for training and evaluating object detection models
on the Kinder Surprise Figures dataset.

Models:
  - YOLOv5s: Fast and accurate YOLO variant with transfer learning
  - MobileNetV3-SSDLite: Lightweight SSD variant for comparison

Usage:
  # Train both models
  uv run python main.py train --all

  # Train specific model
  uv run python main.py train --yolo
  uv run python main.py train --mobilenet

  # Evaluate and compare models
  uv run python main.py evaluate

  # Run inference on an image
  uv run python main.py predict --image path/to/image.jpg

  # Compare inference speed
  uv run python main.py speed
"""

import argparse
import sys
from pathlib import Path


def train_yolo(epochs=100, batch_size=16):
    """Train YOLOv5s model."""
    print("\n" + "=" * 70)
    print("TRAINING YOLOv5s")
    print("=" * 70 + "\n")
    
    from train_yolo import train
    import train_yolo
    
    train_yolo.EPOCHS = epochs
    train_yolo.BATCH_SIZE = batch_size
    
    return train()


def train_mobilenet_model(epochs=50, batch_size=8):
    """Train MobileNetV3-SSDLite model."""
    print("\n" + "=" * 70)
    print("TRAINING MobileNetV3-SSDLite")
    print("=" * 70 + "\n")
    
    import train_mobilenet
    
    train_mobilenet.NUM_EPOCHS = epochs
    train_mobilenet.BATCH_SIZE = batch_size
    
    return train_mobilenet.train()


def evaluate_models():
    """Evaluate and compare both models."""
    from evaluate import compare_models
    return compare_models()


def predict_image(image_path, model='yolo'):
    """Run inference on an image."""
    if model == 'yolo':
        from train_yolo import predict
        predict(image_path)
    elif model == 'mobilenet':
        from train_mobilenet import predict
        predict(image_path)
    else:
        # Run both and compare
        from evaluate import visualize_predictions
        visualize_predictions(image_path)


def run_speed_test():
    """Run inference speed comparison."""
    from evaluate import inference_speed_test
    return inference_speed_test()


def check_dataset():
    """Verify dataset is properly set up."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "dataset"
    
    print("\n" + "=" * 70)
    print("DATASET CHECK")
    print("=" * 70)
    
    # Check directories exist
    train_images = list((data_dir / "images" / "train").glob("*.jpg"))
    val_images = list((data_dir / "images" / "val").glob("*.jpg"))
    train_labels = list((data_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((data_dir / "labels" / "val").glob("*.txt"))
    
    print(f"\nDataset directory: {data_dir}")
    print(f"  Training images:  {len(train_images)}")
    print(f"  Training labels:  {len(train_labels)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Validation labels: {len(val_labels)}")
    
    # Check data.yaml
    data_yaml = data_dir / "data.yaml"
    if data_yaml.exists():
        print("  data.yaml: OK")
    else:
        print("  data.yaml: MISSING")
        return False
    
    # Check for class distribution
    all_labels = train_labels + val_labels
    class_counts = {}
    for label_file in all_labels:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\nClass distribution:")
    class_names = ['penguin_figure', 'octopus_figure', 'battery', 'controller']
    for cls, count in sorted(class_counts.items()):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"  {cls}: {name} - {count} instances")
    
    print("\n" + "=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Object Detection Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both models
  python main.py train --all

  # Train only YOLOv5
  python main.py train --yolo --epochs 100 --batch 16

  # Evaluate trained models
  python main.py evaluate

  # Run prediction
  python main.py predict --image dataset/images/val/IMG_0436.jpg

  # Check dataset
  python main.py check
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--all", action="store_true", help="Train both models")
    train_parser.add_argument("--yolo", action="store_true", help="Train YOLOv5s")
    train_parser.add_argument("--mobilenet", action="store_true", help="Train MobileNet-SSD")
    train_parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=None, help="Batch size")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate and compare models")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on image")
    predict_parser.add_argument("--image", type=str, required=True, help="Image path")
    predict_parser.add_argument("--model", type=str, default="both", 
                                choices=["yolo", "mobilenet", "both"],
                                help="Model to use for prediction")
    
    # Speed test command
    speed_parser = subparsers.add_parser("speed", help="Run inference speed test")
    
    # Check dataset command
    check_parser = subparsers.add_parser("check", help="Check dataset setup")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "train":
        if not check_dataset():
            print("Dataset check failed. Please ensure your dataset is properly set up.")
            sys.exit(1)
        
        if args.all or (not args.yolo and not args.mobilenet):
            # Train both
            yolo_epochs = args.epochs if args.epochs else 100
            yolo_batch = args.batch if args.batch else 16
            train_yolo(epochs=yolo_epochs, batch_size=yolo_batch)
            
            mobilenet_epochs = args.epochs if args.epochs else 50
            mobilenet_batch = args.batch if args.batch else 8
            train_mobilenet_model(epochs=mobilenet_epochs, batch_size=mobilenet_batch)
        else:
            if args.yolo:
                epochs = args.epochs if args.epochs else 100
                batch = args.batch if args.batch else 16
                train_yolo(epochs=epochs, batch_size=batch)
            
            if args.mobilenet:
                epochs = args.epochs if args.epochs else 50
                batch = args.batch if args.batch else 8
                train_mobilenet_model(epochs=epochs, batch_size=batch)
    
    elif args.command == "evaluate":
        evaluate_models()
    
    elif args.command == "predict":
        predict_image(args.image, args.model)
    
    elif args.command == "speed":
        run_speed_test()
    
    elif args.command == "check":
        check_dataset()


if __name__ == "__main__":
    main()
