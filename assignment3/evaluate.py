"""
Evaluation and Comparison Script for Object Detection Models

This script evaluates both YOLOv5s and MobileNetV3-SSDLite models and compares their performance.
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"
YOLO_WEIGHTS = BASE_DIR / "runs" / "yolo" / "yolov5s_kinder" / "weights" / "best.pt"
MOBILENET_WEIGHTS = BASE_DIR / "runs" / "mobilenet" / "mobilenet_ssd_best.pth"

# Class names
CLASS_NAMES = ['penguin_figure', 'octopus_figure', 'battery', 'controller']


def evaluate_yolo(weights_path=None):
    """Evaluate YOLOv5 model and return metrics."""
    from ultralytics import YOLO
    
    if weights_path is None:
        weights_path = YOLO_WEIGHTS
    
    if not Path(weights_path).exists():
        print(f"YOLO weights not found at {weights_path}")
        return None
    
    print("Evaluating YOLOv5s...")
    model = YOLO(str(weights_path))
    
    results = model.val(
        data=str(DATA_DIR / "data.yaml"),
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
    }
    
    # Calculate F1 from precision and recall
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0
    
    return metrics


def evaluate_mobilenet(weights_path=None):
    """Evaluate MobileNet-SSD model and return metrics."""
    from train_mobilenet import evaluate
    
    if weights_path is None:
        weights_path = MOBILENET_WEIGHTS
    
    if not Path(weights_path).exists():
        print(f"MobileNet weights not found at {weights_path}")
        return None
    
    print("Evaluating MobileNetV3-SSDLite...")
    metrics = evaluate(weights_path)
    
    return metrics


def inference_speed_test(num_runs=50):
    """Compare inference speed of both models."""
    from ultralytics import YOLO
    from train_mobilenet import get_model, val_transform, NUM_CLASSES, DEVICE
    
    # Test image
    test_images = list((DATA_DIR / "images" / "val").glob("*.jpg"))
    if not test_images:
        print("No validation images found for speed test")
        return None
    
    test_image = test_images[0]
    print(f"\nRunning inference speed test on {test_image.name} ({num_runs} iterations)...")
    
    results = {}
    
    # YOLOv5 speed test
    if YOLO_WEIGHTS.exists():
        yolo_model = YOLO(str(YOLO_WEIGHTS))
        
        # Warmup
        for _ in range(5):
            _ = yolo_model(str(test_image), verbose=False)
        
        # Timed runs
        start = time.time()
        for _ in range(num_runs):
            _ = yolo_model(str(test_image), verbose=False)
        yolo_time = (time.time() - start) / num_runs * 1000  # ms per image
        results['YOLOv5s'] = yolo_time
        print(f"YOLOv5s: {yolo_time:.2f} ms/image")
    
    # MobileNet speed test
    if MOBILENET_WEIGHTS.exists():
        mobilenet_model = get_model(NUM_CLASSES)
        checkpoint = torch.load(MOBILENET_WEIGHTS, map_location=DEVICE)
        mobilenet_model.load_state_dict(checkpoint['model_state_dict'])
        mobilenet_model.to(DEVICE)
        mobilenet_model.eval()
        
        # Prepare image
        image = np.array(Image.open(test_image).convert("RGB"))
        transformed = val_transform(image=image, bboxes=[], class_labels=[])
        input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = mobilenet_model(input_tensor)
        
        # Timed runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = mobilenet_model(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        mobilenet_time = (time.time() - start) / num_runs * 1000  # ms per image
        results['MobileNet-SSD'] = mobilenet_time
        print(f"MobileNetV3-SSDLite: {mobilenet_time:.2f} ms/image")
    
    return results


def compare_models():
    """Compare both models and generate a summary report."""
    print("=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)
    
    # Evaluate models
    yolo_metrics = evaluate_yolo()
    mobilenet_metrics = evaluate_mobilenet()
    
    # Speed test
    speed_results = inference_speed_test()
    
    # Print comparison table
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<20} {'YOLOv5s':<20} {'MobileNet-SSD':<20}")
    print("-" * 70)
    
    if yolo_metrics and mobilenet_metrics:
        print(f"{'Precision':<20} {yolo_metrics['precision']:.4f}{'':<14} {mobilenet_metrics['precision']:.4f}")
        print(f"{'Recall':<20} {yolo_metrics['recall']:.4f}{'':<14} {mobilenet_metrics['recall']:.4f}")
        print(f"{'F1-Score':<20} {yolo_metrics['f1']:.4f}{'':<14} {mobilenet_metrics['f1']:.4f}")
        
        if 'mAP50' in yolo_metrics:
            print(f"{'mAP@0.5':<20} {yolo_metrics['mAP50']:.4f}{'':<14} {'N/A'}")
            print(f"{'mAP@0.5:0.95':<20} {yolo_metrics['mAP50-95']:.4f}{'':<14} {'N/A'}")
    
    if speed_results:
        print("-" * 70)
        print("INFERENCE SPEED")
        print("-" * 70)
        for model, speed in speed_results.items():
            print(f"{model:<20} {speed:.2f} ms/image")
    
    print("-" * 70)
    
    # Generate comparison plot
    if yolo_metrics and mobilenet_metrics:
        generate_comparison_plot(yolo_metrics, mobilenet_metrics)
    
    return yolo_metrics, mobilenet_metrics, speed_results


def generate_comparison_plot(yolo_metrics, mobilenet_metrics):
    """Generate a bar chart comparing model metrics."""
    metrics = ['precision', 'recall', 'f1']
    yolo_values = [yolo_metrics.get(m, 0) for m in metrics]
    mobilenet_values = [mobilenet_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    _, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, yolo_values, width, label='YOLOv5s', color='#3498db')
    bars2 = ax.bar(x + width/2, mobilenet_values, width, label='MobileNet-SSD', color='#e74c3c')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {BASE_DIR / 'model_comparison.png'}")
    plt.close()


def visualize_predictions(image_path, save_path=None):
    """Visualize predictions from both models side by side."""
    from ultralytics import YOLO
    from train_mobilenet import get_model, val_transform, NUM_CLASSES, DEVICE
    import matplotlib.patches as patches
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    
    # Load original image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    _, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # YOLOv5 prediction
    if YOLO_WEIGHTS.exists():
        yolo_model = YOLO(str(YOLO_WEIGHTS))
        yolo_results = yolo_model(str(image_path), verbose=False)[0]
        
        axes[0].imshow(image_np)
        axes[0].set_title('YOLOv5s Detection', fontsize=14)
        
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            color = colors[cls % len(colors)]
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[0].add_patch(rect)
            
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
            axes[0].text(x1, y1-5, f'{label}: {conf:.2f}',
                        color='white', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'YOLOv5 weights not found', 
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # MobileNet-SSD prediction
    if MOBILENET_WEIGHTS.exists():
        mobilenet_model = get_model(NUM_CLASSES)
        checkpoint = torch.load(MOBILENET_WEIGHTS, map_location=DEVICE)
        mobilenet_model.load_state_dict(checkpoint['model_state_dict'])
        mobilenet_model.to(DEVICE)
        mobilenet_model.eval()
        
        # Transform image
        transformed = val_transform(image=image_np, bboxes=[], class_labels=[])
        input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = mobilenet_model(input_tensor)[0]
        
        axes[1].imshow(image_np)
        axes[1].set_title('MobileNetV3-SSDLite Detection', fontsize=14)
        
        # Scale boxes
        scale_x = image_np.shape[1] / 320
        scale_y = image_np.shape[0] / 320
        
        mask = outputs['scores'] > 0.5
        boxes = outputs['boxes'][mask].cpu().numpy()
        labels = outputs['labels'][mask].cpu().numpy()
        scores = outputs['scores'][mask].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            # Label is 1-indexed (0 is background)
            cls = label - 1 if label > 0 else 0
            color = colors[cls % len(colors)]
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[1].add_patch(rect)
            
            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
            axes[1].text(x1, y1-5, f'{class_name}: {score:.2f}',
                        color='white', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'MobileNet weights not found', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
    
    plt.suptitle(f'Detection Comparison: {image_path.name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and Compare Object Detection Models")
    parser.add_argument("--mode", type=str, default="compare", 
                        choices=["compare", "yolo", "mobilenet", "visualize", "speed"],
                        help="Evaluation mode")
    parser.add_argument("--image", type=str, default=None, help="Image path for visualization")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_models()
    elif args.mode == "yolo":
        metrics = evaluate_yolo()
        if metrics:
            print("\nYOLOv5s Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
    elif args.mode == "mobilenet":
        metrics = evaluate_mobilenet()
        if metrics:
            print("\nMobileNet-SSD Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
    elif args.mode == "visualize":
        if args.image:
            visualize_predictions(args.image)
        else:
            # Use first validation image
            val_images = list((DATA_DIR / "images" / "val").glob("*.jpg"))
            if val_images:
                visualize_predictions(val_images[0])
            else:
                print("No validation images found")
    elif args.mode == "speed":
        inference_speed_test()
