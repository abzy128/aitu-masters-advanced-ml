# Object Detection Project: Kinder Surprise Figures Dataset

## Project Overview

This document outlines a complete plan for creating a custom object detection system using Kinder Surprise toy figures as the dataset. The project is designed to be completed in 1-2 days with limited computational resources.

| Parameter | Value |
|-----------|-------|
| Dataset | Custom Kinder Surprise figures |
| Classes | 3-5 different figures |
| Total Images | ~150-200 |
| Models | YOLOv5s + MobileNetV2-SSD |
| Hardware | RTX 4060 8GB |
| Timeline | 1-2 days |

---

## Part 1: Dataset Creation

### 1.1 Equipment and Image Specifications

**Your camera setup:**
- Phone camera with macro capability
- Output: 3024×3024 HEIF files

**Image preprocessing needed:**
- Convert HEIF → JPEG (required for training)
- Resize to 640×640 (YOLOv5 default)

**Conversion script (run once on all images):**

```bash
# Install ImageMagick if not present
sudo apt install imagemagick

# Convert all HEIF files to JPEG and resize
mkdir -p converted_images
for f in *.heif *.HEIF; do
    [ -f "$f" ] || continue
    convert "$f" -resize 640x640 "converted_images/${f%.*}.jpg"
done
```

Or in Python:

```python
from PIL import Image
from pillow_heif import register_heif_opener
import os

register_heif_opener()

input_dir = "raw_photos"
output_dir = "converted_images"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.heif', '.heic')):
        img = Image.open(os.path.join(input_dir, filename))
        img = img.convert("RGB")
        img = img.resize((640, 640))
        output_name = os.path.splitext(filename)[0] + ".jpg"
        img.save(os.path.join(output_dir, output_name), "JPEG", quality=95)
        print(f"Converted: {filename}")
```

**Required package:**
```bash
pip install pillow-heif
```

### 1.2 Photography Guidelines

**Collect 3-5 Kinder figures. For each figure, take approximately 40-50 photos:**

| Shot Type | Count | Description |
|-----------|-------|-------------|
| Plain background | 10 | White paper, solid color surface |
| Textured background | 10 | Wooden table, fabric, carpet |
| With other figures | 10 | 2-3 figures in same frame |
| Different angles | 10 | Front, back, side, top, 45° |
| Varied lighting | 5-10 | Natural light, lamp, shadows |

**Tips for better results:**
- Keep figures in focus (use macro mode)
- Avoid motion blur
- Include some partially visible figures (occlusion)
- Vary the distance from camera
- Rotate the figure, not just the camera

**Total target: ~150-200 images**

### 1.3 Annotation with MakeSense.ai

**Step-by-step annotation process:**

1. Go to [makesense.ai](https://www.makesense.ai/) (free, no account needed)

2. Click "Get Started" → Drop your converted JPEG images

3. Select "Object Detection"

4. Create labels for each figure class:
   - Example: `figure_lion`, `figure_penguin`, `figure_robot`, etc.
   - Use simple, descriptive names (no spaces, use underscores)

5. For each image:
   - Draw bounding box around each figure
   - Select the correct class label
   - Box should be tight but include entire figure

6. Export annotations:
   - Click "Actions" → "Export Annotations"
   - Select **"YOLO format"** (creates .txt files)
   - Download the ZIP file

### 1.4 Dataset Structure

After annotation, organize your files like this:

```
kinder_dataset/
├── images/
│   ├── train/          # ~70% of images (105-140 images)
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/            # ~30% of images (45-60 images)
│       ├── img_150.jpg
│       └── ...
├── labels/
│   ├── train/          # Corresponding .txt files
│   │   ├── img_001.txt
│   │   └── ...
│   └── val/
│       ├── img_150.txt
│       └── ...
└── data.yaml           # Dataset configuration
```

**Python script to split dataset:**

```python
import os
import shutil
import random

# Configuration
source_images = "converted_images"
source_labels = "annotations"  # Exported from MakeSense.ai
output_dir = "kinder_dataset"
train_ratio = 0.7

# Create directories
for split in ['train', 'val']:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

# Get all image files
images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
random.shuffle(images)

# Split
split_idx = int(len(images) * train_ratio)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Copy files
for img_list, split in [(train_images, 'train'), (val_images, 'val')]:
    for img in img_list:
        # Copy image
        shutil.copy(
            f"{source_images}/{img}",
            f"{output_dir}/images/{split}/{img}"
        )
        # Copy label
        label = img.replace('.jpg', '.txt')
        if os.path.exists(f"{source_labels}/{label}"):
            shutil.copy(
                f"{source_labels}/{label}",
                f"{output_dir}/labels/{split}/{label}"
            )

print(f"Train: {len(train_images)} images")
print(f"Val: {len(val_images)} images")
```

**Create `data.yaml` configuration file:**

```yaml
# kinder_dataset/data.yaml
path: /path/to/kinder_dataset  # Update this path!
train: images/train
val: images/val

# Number of classes
nc: 4  # Update based on your figure count

# Class names (must match annotation labels exactly)
names:
  0: figure_lion
  1: figure_penguin
  2: figure_robot
  3: figure_dinosaur
```

---

## Part 2: Model Training

### 2.1 Environment Setup

```bash
# Create project directory
mkdir kinder_detection
cd kinder_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch with CUDA (for RTX 4060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# Additional packages
pip install albumentations  # For MobileNet augmentation
```

### 2.2 Model 1: YOLOv5s with Transfer Learning

**Why YOLOv5s:**
- Smallest YOLO variant (~14MB, 7.2M parameters)
- Fast training and inference
- Pre-trained on COCO dataset (80 classes)
- Perfect for small custom datasets

**Training command:**

```bash
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data /path/to/kinder_dataset/data.yaml \
    --weights yolov5s.pt \
    --project runs/kinder \
    --name yolov5s_kinder \
    --cache
```

**Parameter explanation:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--img` | 640 | Standard YOLOv5 input size |
| `--batch` | 16 | Comfortable for 8GB VRAM (~4GB usage) |
| `--epochs` | 100 | Sufficient for small dataset |
| `--weights` | yolov5s.pt | Pre-trained weights (transfer learning) |
| `--cache` | - | Cache images in RAM for faster training |

**Expected training time:** ~30-60 minutes on RTX 4060

**Transfer learning happens automatically:**
- YOLOv5 loads pre-trained weights from `yolov5s.pt`
- Only the detection head is randomly initialized for your classes
- Backbone features (learned from COCO) are fine-tuned

**Built-in data augmentation (enabled by default):**

YOLOv5 automatically applies these augmentations during training:

| Augmentation | Default Value | Effect |
|--------------|---------------|--------|
| HSV-Hue | 0.015 | Color variation |
| HSV-Saturation | 0.7 | Saturation variation |
| HSV-Value | 0.4 | Brightness variation |
| Rotation | 0.0 | Degrees (increase to 15 if needed) |
| Translation | 0.1 | Shift image |
| Scale | 0.5 | Zoom in/out |
| Fliplr | 0.5 | Horizontal flip (50% chance) |
| Mosaic | 1.0 | Combine 4 images |
| Mixup | 0.0 | Blend images |

**To customize augmentation, create `hyp_custom.yaml`:**

```yaml
# hyp_custom.yaml - Custom hyperparameters
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
# Augmentation settings
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 15.0        # Enable rotation
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.1           # Enable mixup
copy_paste: 0.0
```

**Use custom hyperparameters:**

```bash
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data /path/to/kinder_dataset/data.yaml \
    --weights yolov5s.pt \
    --hyp hyp_custom.yaml \
    --project runs/kinder \
    --name yolov5s_kinder
```

### 2.3 Model 2: MobileNetV2-SSD with Transfer Learning

**Why MobileNetV2-SSD:**
- Lightweight architecture designed for mobile devices
- Good comparison with YOLOv5 (different approach)
- Uses SSD (Single Shot Detector) architecture

**Installation:**

```bash
pip install torchvision
```

**Complete training script with augmentation (`train_mobilenet.py`):**

```python
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np

# ============ Configuration ============
DATA_DIR = "/path/to/kinder_dataset"
NUM_CLASSES = 5  # 4 figures + 1 background
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Data Augmentation ============
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ============ Dataset Class ============
class KinderDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load labels (YOLO format: class x_center y_center width height)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
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
                boxes = transformed['bboxes']
                class_labels = transformed['class_labels']
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
            target_boxes.append([x1, y1, x2, y2])
        
        target = {
            'boxes': torch.tensor(target_boxes, dtype=torch.float32) if target_boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(class_labels, dtype=torch.int64) if class_labels else torch.zeros(0, dtype=torch.int64)
        }
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ============ Model Setup ============
def get_model(num_classes):
    # Load pre-trained MobileNetV3-SSDLite
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    
    # Replace the classification head for our number of classes
    in_channels = model.head.classification_head.module_list[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Create new classification head
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model

# ============ Training Loop ============
def train():
    print(f"Using device: {DEVICE}")
    
    # Datasets
    train_dataset = KinderDataset(
        f"{DATA_DIR}/images/train",
        f"{DATA_DIR}/labels/train",
        transform=train_transform
    )
    val_dataset = KinderDataset(
        f"{DATA_DIR}/images/val",
        f"{DATA_DIR}/labels/val",
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Skip empty batches
            if any(len(t['boxes']) == 0 for t in targets):
                continue
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'mobilenet_ssd_kinder_best.pth')
            print(f"  Saved best model (loss: {best_loss:.4f})")
    
    print("Training complete!")
    torch.save(model.state_dict(), 'mobilenet_ssd_kinder_final.pth')

if __name__ == "__main__":
    train()
```

**Run training:**

```bash
python train_mobilenet.py
```

**Expected training time:** ~20-40 minutes on RTX 4060

---

## Part 3: Evaluation and Comparison

### 3.1 YOLOv5 Validation

```bash
cd yolov5

# Run validation
python val.py \
    --weights runs/kinder/yolov5s_kinder/weights/best.pt \
    --data /path/to/kinder_dataset/data.yaml \
    --img 640 \
    --task test \
    --save-json \
    --save-txt
```

**Output metrics:**
- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- F1-score

### 3.2 MobileNet-SSD Evaluation Script

```python
# evaluate_mobilenet.py
import torch
from torchvision.ops import box_iou
from train_mobilenet import KinderDataset, val_transform, get_model, collate_fn, DATA_DIR, NUM_CLASSES, DEVICE
from torch.utils.data import DataLoader
import numpy as np

def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """Calculate precision, recall, and F1-score."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, target in zip(predictions, targets):
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
                    best_iou = iou_matrix[i, j]
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
    
    return precision, recall, f1

def evaluate():
    # Load model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load('mobilenet_ssd_kinder_best.pth'))
    model.to(DEVICE)
    model.eval()
    
    # Load validation data
    val_dataset = KinderDataset(
        f"{DATA_DIR}/images/val",
        f"{DATA_DIR}/labels/val",
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    predictions = []
    targets_list = []
    
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
                predictions.append(pred)
                targets_list.append({k: v.cpu() for k, v in target.items()})
    
    # Calculate metrics
    precision, recall, f1 = calculate_metrics(predictions, targets_list)
    
    print("\n" + "="*50)
    print("MobileNetV2-SSD Evaluation Results")
    print("="*50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate()
```

### 3.3 Expected Results Comparison

| Metric | YOLOv5s | MobileNetV2-SSD |
|--------|---------|-----------------|
| mAP@0.5 | ~0.85-0.95 | ~0.75-0.85 |
| Precision | ~0.85-0.90 | ~0.80-0.85 |
| Recall | ~0.80-0.90 | ~0.75-0.85 |
| F1-Score | ~0.85-0.90 | ~0.78-0.85 |
| Inference (ms) | ~5-10 | ~8-15 |
| Model Size | ~14 MB | ~12 MB |

---

## Part 4: Inference and Visualization

### 4.1 YOLOv5 Inference

```bash
# Single image
python detect.py \
    --weights runs/kinder/yolov5s_kinder/weights/best.pt \
    --source /path/to/test_image.jpg \
    --conf-thres 0.5

# Folder of images
python detect.py \
    --weights runs/kinder/yolov5s_kinder/weights/best.pt \
    --source /path/to/test_images/ \
    --conf-thres 0.5

# Webcam (real-time)
python detect.py \
    --weights runs/kinder/yolov5s_kinder/weights/best.pt \
    --source 0 \
    --conf-thres 0.5
```

### 4.2 MobileNet-SSD Inference Script

```python
# inference_mobilenet.py
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train_mobilenet import get_model, val_transform, NUM_CLASSES, DEVICE
import numpy as np

# Class names (update to match your figures)
CLASS_NAMES = ['background', 'figure_lion', 'figure_penguin', 'figure_robot', 'figure_dinosaur']
COLORS = ['red', 'blue', 'green', 'orange', 'purple']

def detect(image_path, model_path='mobilenet_ssd_kinder_best.pth', conf_threshold=0.5):
    # Load model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
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
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)
    
    # Scale boxes to original image size
    h, w = image_np.shape[:2]
    scale_x = w / 320
    scale_y = h / 320
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        
        color = COLORS[label % len(COLORS)]
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
    plt.savefig('detection_result.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Detected {len(boxes)} objects")
    return boxes, labels, scores

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    detect(image_path)
```

**Usage:**

```bash
python inference_mobilenet.py test_image.jpg
```

### 4.3 Side-by-Side Comparison Visualization

```python
# compare_models.py
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def compare_detections(image_path):
    # Run YOLOv5 detection
    subprocess.run([
        'python', 'yolov5/detect.py',
        '--weights', 'runs/kinder/yolov5s_kinder/weights/best.pt',
        '--source', image_path,
        '--save-txt', '--save-conf',
        '--project', 'comparison', '--name', 'yolo',
        '--exist-ok'
    ])
    
    # Run MobileNet detection (generates detection_result.jpg)
    from inference_mobilenet import detect
    detect(image_path)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # YOLOv5 result
    yolo_result = f'comparison/yolo/{image_path.split("/")[-1]}'
    axes[0].imshow(mpimg.imread(yolo_result))
    axes[0].set_title('YOLOv5s Detection', fontsize=14)
    axes[0].axis('off')
    
    # MobileNet result
    axes[1].imshow(mpimg.imread('detection_result.jpg'))
    axes[1].set_title('MobileNetV2-SSD Detection', fontsize=14)
    axes[1].axis('off')
    
    plt.suptitle('Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_comparison.jpg', dpi=150)
    plt.show()

if __name__ == "__main__":
    import sys
    compare_detections(sys.argv[1])
```

---

## Part 5: Project Timeline

### Day 1 (4-6 hours)

| Time | Task |
|------|------|
| 1 hour | Photography (150-200 images) |
| 1 hour | Convert HEIF → JPEG, organize files |
| 1-2 hours | Annotate images in MakeSense.ai |
| 1 hour | Split dataset, create data.yaml |
| 30 min | Set up environment |

### Day 2 (3-5 hours)

| Time | Task |
|------|------|
| 1 hour | Train YOLOv5s model |
| 1 hour | Train MobileNetV2-SSD model |
| 1 hour | Run evaluation, collect metrics |
| 1-2 hours | Create visualizations, write report |

---

## Part 6: Report Structure Checklist

Based on assignment requirements:

- [ ] **1. Dataset Description**
  - Source: Custom Kinder Surprise figures
  - Size: X images, Y classes
  - Split: 70/30 train/val
  - Link: N/A (custom dataset)

- [ ] **2. Theoretical Background**
  - [ ] Neural Networks for Object Recognition (CNNs)
  - [ ] Validation and Cross-Validation explanation
  - [ ] Overview of 8 architectures (from assignment PDF)

- [ ] **3. Implementation**
  - [ ] Data preprocessing pipeline
  - [ ] YOLOv5s training with transfer learning
  - [ ] MobileNetV2-SSD training with transfer learning
  - [ ] Data augmentation methods used

- [ ] **4. Experimental Results**
  - [ ] Metrics table (Precision, Recall, F1, mAP)
  - [ ] Training curves (loss, mAP over epochs)
  - [ ] Detection visualizations with bounding boxes
  - [ ] Model comparison analysis

---

## Quick Reference Commands

```bash
# Convert images
python convert_heif.py

# Split dataset
python split_dataset.py

# Train YOLOv5
cd yolov5 && python train.py --img 640 --batch 16 --epochs 100 \
    --data ../kinder_dataset/data.yaml --weights yolov5s.pt

# Train MobileNet
python train_mobilenet.py

# Evaluate YOLOv5
python val.py --weights runs/kinder/yolov5s_kinder/weights/best.pt \
    --data ../kinder_dataset/data.yaml

# Evaluate MobileNet
python evaluate_mobilenet.py

# Run inference
python detect.py --weights best.pt --source test.jpg  # YOLOv5
python inference_mobilenet.py test.jpg                 # MobileNet
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size to 8 or 4 |
| HEIF not recognized | Install `pillow-heif` package |
| Empty annotations | Check MakeSense.ai export format is YOLO |
| Low accuracy | Add more images, check annotation quality |
| Slow training | Enable `--cache` flag, verify GPU is being used |

---

## Resources

- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
- [MakeSense.ai](https://www.makesense.ai/)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [PyTorch Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)