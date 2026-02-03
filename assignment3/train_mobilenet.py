"""
MobileNetV3-SSDLite Training Script for Kinder Surprise Figures Detection

This script trains a MobileNetV3-SSDLite model with transfer learning.
Uses pre-trained COCO weights for the backbone.
"""

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from pathlib import Path
import time

# ============ Configuration ============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "runs" / "mobilenet"

NUM_CLASSES = 5  # 4 figures + 1 background
BATCH_SIZE = 8   # SSDLite uses smaller images, can use larger batch
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names for reference
CLASS_NAMES = ['background', 'penguin_figure', 'octopus_figure', 'battery', 'controller']

# ============ Data Augmentation ============
train_transform = A.Compose([
    A.Resize(320, 320),  # SSDLite input size
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Rotate(limit=15, p=0.3, border_mode=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

val_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


# ============ Dataset Class ============
class KinderDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = self.img_dir / img_name
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load labels (YOLO format: class x_center y_center width height)
        label_path = self.label_dir / img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        boxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        # Clamp values to valid range
                        x_center = max(0.001, min(0.999, x_center))
                        y_center = max(0.001, min(0.999, y_center))
                        w = max(0.001, min(0.999, w))
                        h = max(0.001, min(0.999, h))
                        boxes.append([x_center, y_center, w, h])
                        class_labels.append(class_id + 1)  # +1 for background class
        
        # Apply transforms
        if self.transform:
            if len(boxes) > 0:
                transformed = self.transform(
                    image=image, 
                    bboxes=boxes, 
                    class_labels=class_labels
                )
                image = transformed['image']
                boxes = list(transformed['bboxes'])
                class_labels = list(transformed['class_labels'])
            else:
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
                image = transformed['image']
        
        # Convert YOLO format to absolute coordinates for PyTorch
        h, w = 320, 320  # SSDLite input size
        target_boxes = []
        for box in boxes:
            x_center, y_center, bw, bh = box
            x1 = (x_center - bw/2) * w
            y1 = (y_center - bh/2) * h
            x2 = (x_center + bw/2) * w
            y2 = (y_center + bh/2) * h
            # Ensure valid boxes
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                target_boxes.append([x1, y1, x2, y2])
        
        # Filter out invalid boxes
        valid_indices = list(range(len(target_boxes)))
        class_labels = [class_labels[i] for i in valid_indices if i < len(class_labels)]
        
        target = {
            'boxes': torch.tensor(target_boxes, dtype=torch.float32) if target_boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(class_labels, dtype=torch.int64) if class_labels else torch.zeros(0, dtype=torch.int64)
        }
        
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============ Model Setup ============
def get_model(num_classes):
    """Load pre-trained MobileNetV3-SSDLite and modify for custom classes."""
    # Load pre-trained model (transfer learning)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    
    # Get the number of input channels and anchors for the classification head
    in_channels = [672, 480, 512, 256, 256, 128]
    num_anchors = [6, 6, 6, 6, 6, 6]
    
    # Replace the classification head for our number of classes
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model


# ============ Training Loop ============
def train():
    print("=" * 60)
    print("MobileNetV3-SSDLite Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of classes: {NUM_CLASSES} (including background)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("-" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Datasets
    train_dataset = KinderDataset(
        DATA_DIR / "images" / "train",
        DATA_DIR / "labels" / "train",
        transform=train_transform
    )
    val_dataset = KinderDataset(
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "val",
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print("-" * 60)
    
    # Model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0
        start_time = time.time()
        
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Skip batches with no valid boxes
            valid_targets = [t for t in targets if len(t['boxes']) > 0]
            if len(valid_targets) == 0:
                continue
            
            # Filter images to match valid targets
            valid_indices = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
            images = [images[i] for i in valid_indices]
            targets = valid_targets
            
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()
                
                epoch_loss += losses.item()
                batch_count += 1
            except (RuntimeError, ValueError) as e:
                print(f"Warning: Batch skipped due to error: {e}")
                continue
        
        lr_scheduler.step()
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            epoch_time = time.time() - start_time
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f} - "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} - Time: {epoch_time:.1f}s")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'class_names': CLASS_NAMES,
                }, OUTPUT_DIR / 'mobilenet_ssd_best.pth')
                print(f"  -> Saved best model (loss: {best_loss:.4f})")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss if batch_count > 0 else float('inf'),
        'class_names': CLASS_NAMES,
        'train_losses': train_losses,
    }, OUTPUT_DIR / 'mobilenet_ssd_final.pth')
    
    print(f"Best model: {OUTPUT_DIR / 'mobilenet_ssd_best.pth'}")
    print(f"Final model: {OUTPUT_DIR / 'mobilenet_ssd_final.pth'}")
    
    return train_losses


# ============ Evaluation ============
def evaluate(weights_path=None):
    """Evaluate the trained model on validation set."""
    from torchvision.ops import box_iou
    
    if weights_path is None:
        weights_path = OUTPUT_DIR / 'mobilenet_ssd_best.pth'
    
    print(f"Loading model from: {weights_path}")
    
    # Load model
    model = get_model(NUM_CLASSES)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Load validation data
    val_dataset = KinderDataset(
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "val",
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    all_predictions = []
    all_targets = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                # Filter predictions with confidence > 0.5
                mask = output['scores'] > 0.5
                pred = {
                    'boxes': output['boxes'][mask].cpu(),
                    'labels': output['labels'][mask].cpu(),
                    'scores': output['scores'][mask].cpu()
                }
                all_predictions.append(pred)
                all_targets.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()})
    
    # Calculate metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = 0.5
    
    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            continue
        elif len(pred_boxes) == 0:
            total_fn += len(target_boxes)
            continue
        elif len(target_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        # Calculate IoU
        iou_matrix = box_iou(pred_boxes, target_boxes)
        
        # Match predictions to ground truth
        matched_gt = set()
        for i, pred_label in enumerate(pred_labels):
            best_iou = 0
            best_j = -1
            for j, target_label in enumerate(target_labels):
                if j in matched_gt:
                    continue
                if pred_label == target_label and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j].item()
                    best_j = j
            
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_j)
            else:
                total_fp += 1
        
        total_fn += len(target_boxes) - len(matched_gt)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 60)
    print("MobileNetV3-SSDLite Evaluation Results")
    print("=" * 60)
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print("=" * 60)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


# ============ Inference ============
def predict(image_path, weights_path=None, conf_threshold=0.5, save_path=None):
    """Run inference on a single image."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    if weights_path is None:
        weights_path = OUTPUT_DIR / 'mobilenet_ssd_best.pth'
    
    # Load model
    model = get_model(NUM_CLASSES)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_np = np.array(image)
    
    # Transform for model
    transformed = val_transform(image=image_np, bboxes=[], class_labels=[])
    input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Process outputs
    output = outputs[0]
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Visualize
    _, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)
    
    # Scale boxes to original image size
    scale_x = original_size[0] / 320
    scale_y = original_size[1] / 320
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        
        color = colors[label % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'class_{label}'
        ax.text(x1, y1-5, f'{class_name}: {score:.2f}',
                color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction to: {save_path}")
    
    plt.show()
    
    print(f"Detected {len(boxes)} objects")
    return boxes, labels, scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MobileNetV3-SSDLite Training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"],
                        help="Mode: train, eval, or predict")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file")
    parser.add_argument("--source", type=str, default=None, help="Image path for prediction")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        NUM_EPOCHS = args.epochs
        BATCH_SIZE = args.batch
        train()
    elif args.mode == "eval":
        evaluate(args.weights)
    elif args.mode == "predict":
        if args.source is None:
            print("Error: --source required for prediction mode")
        else:
            predict(args.source, args.weights)
