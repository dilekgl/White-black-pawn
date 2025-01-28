import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import time
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou

# COCO formatındaki bounding box'ları Pascal VOC formatına dönüştürme
def coco_to_pascal_voc(bbox):
    """
    COCO formatı [x_min, y_min, width, height] -> Pascal VOC formatı [x_min, y_min, x_max, y_max].
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

# Bounding box'ların geçerliliğini kontrol etme
def validate_bbox(bbox):
    """
    Bounding box'ın genişlik ve yüksekliğinin pozitif olduğunu kontrol et.
    """
    x_min, y_min, width, height = bbox
    return width > 0 and height > 0

# Bounding box'ların görüntü sınırları içinde olduğunu kontrol etme
def validate_bbox_within_image(bbox, image_width, image_height):
    """
    Bounding box'ın görüntü sınırları içinde olduğunu kontrol et.
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_min >= 0 and y_min >= 0 and x_max <= image_width and y_max <= image_height)

# DataLoader için collate fonksiyonu
def collate_fn(batch):
    return tuple(zip(*batch))

# COCO veri setini yükleme ve filtreleme
def get_coco_dataloader(json_path, img_dir, batch_size=4, shuffle=True):
    dataset = CocoDetection(img_dir, json_path, transform=F.to_tensor)
    
    # Geçersiz bounding box'ları olan görüntüleri filtrele
    valid_indices = []
    for idx in range(len(dataset)):
        _, targets = dataset[idx]
        has_valid_boxes = False
        for annotation in targets:
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]
            if category_id in category_id_mapping and validate_bbox(bbox):
                has_valid_boxes = True
                break
        if has_valid_boxes:
            valid_indices.append(idx)
    
    # Sadece geçerli görüntüleri içeren bir alt küme oluştur
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    
    return DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Kategori ID'lerini eşleme
category_id_mapping = {
    5: 0,  # Kategori ID 5 -> 0
    11: 1, # Kategori ID 11 -> 1
}

# Veri setlerini yükleme
train_loader = get_coco_dataloader("chess-pieces-coco/train/_annotations.coco.json", "fastrcnn/chess-pieces-fastrcnn/train/images")
val_loader = get_coco_dataloader("chess-pieces-coco/valid/_annotations.coco.json", "fastrcnn/chess-pieces-fastrcnn/valid/images")

# Cihazı ayarla (GPU veya CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yükleme
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

# Optimizasyon ve loss fonksiyonu
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Model eğitimi
epochs = 10 # Epoch sayısını artır
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        
        # Her görüntü için hedefleri işle
        processed_targets = []
        for image, image_targets in zip(images, targets):
            image_width, image_height = image.shape[2], image.shape[1]  # Görüntü boyutlarını al
            boxes = []
            labels = []
            image_ids = []
            
            for annotation in image_targets:
                bbox = annotation["bbox"]
                category_id = annotation["category_id"]
                
                # Eşlemede olmayan kategori ID'lerini atla
                if category_id not in category_id_mapping:
                    continue
                
                # Bounding box'ı doğrula
                if not validate_bbox(bbox):
                    print(f"Warning: Invalid bounding box {bbox} found in image ID {annotation['image_id']} and skipped.")
                    continue
                
                # COCO formatını Pascal VOC formatına dönüştür
                bbox_pascal_voc = coco_to_pascal_voc(bbox)
                
                # Bounding box'ın görüntü sınırları içinde olduğunu doğrula
                if not validate_bbox_within_image(bbox_pascal_voc, image_width, image_height):
                    print(f"Warning: Bounding box {bbox_pascal_voc} is outside image dimensions {image_width}x{image_height}. Skipping.")
                    continue
                
                # Kategori ID'sini eşle
                mapped_category_id = category_id_mapping[category_id]
                
                boxes.append(bbox_pascal_voc)
                labels.append(mapped_category_id)
                image_ids.append(annotation["image_id"])
            
            processed_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                "image_id": torch.tensor(image_ids[0], dtype=torch.int64).to(device),  # Use the first image_id
            })
        
        # Forward pass
        optimizer.zero_grad()
        loss_dict = model(images, processed_targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

# Modeli değerlendirme
model.eval()
y_true = []
y_pred = []
inference_times = []
detections = []

with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        
        # Process targets for each image in the batch
        processed_targets = []
        for image_targets in targets:
            boxes = []
            labels = []
            image_ids = []
            
            for annotation in image_targets:
                bbox = annotation["bbox"]
                category_id = annotation["category_id"]
                
                # Skip objects with category_id not in the mapping
                if category_id not in category_id_mapping:
                    continue
                
                # Validate bounding box
                if not validate_bbox(bbox):
                    print(f"Warning: Invalid bounding box {bbox} found in image ID {annotation['image_id']} and skipped.")
                    continue
                
                # Convert COCO format to Pascal VOC format
                bbox_pascal_voc = coco_to_pascal_voc(bbox)
                
                # Map category_id to new values
                mapped_category_id = category_id_mapping[category_id]
                
                boxes.append(bbox_pascal_voc)
                labels.append(mapped_category_id)
                image_ids.append(annotation["image_id"])
            
            processed_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                "image_id": torch.tensor(image_ids[0], dtype=torch.int64).to(device),  # Use the first image_id
            })
        
        # Forward pass
        start_time = time.time()
        outputs = model(images)
        inference_times.append(time.time() - start_time)
        
        # Process outputs and collect predictions
        for output, target in zip(outputs, processed_targets):
            pred_boxes = output['boxes']
            pred_labels = output['labels']
            pred_scores = output['scores']
            true_boxes = target['boxes']
            true_labels = target['labels']

            # Calculate IoU between predicted and ground truth boxes
            iou_matrix = box_iou(pred_boxes, true_boxes)
            matched_indices = iou_matrix.max(dim=1).indices

            # Filter predictions based on IoU threshold
            iou_threshold = 0.5
            for i, (pred_label, pred_score, matched_idx) in enumerate(zip(pred_labels, pred_scores, matched_indices)):
                if iou_matrix[i, matched_idx] >= iou_threshold:
                    y_pred.append(pred_label.item())
                    y_true.append(true_labels[matched_idx].item())

            # Convert predictions to COCO format
            for j in range(len(pred_boxes)):
                box = pred_boxes[j].cpu().numpy().tolist()
                score = pred_scores[j].item()
                category_id = pred_labels[j].item()
                detections.append({
                    "image_id": int(target['image_id'].item()),  # Ensure image_id is a scalar
                    "category_id": category_id,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": score,
                })

# Confusion Matrix hesaplama
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved.")

# Ortalama inference süresi
avg_inference_time = np.mean(inference_times)
print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

# Model boyutu hesaplama
torch.save(model.state_dict(), "fastrcnn_model.pth")
model_size = os.path.getsize("fastrcnn_model.pth") / (1024 * 1024)
print(f"Model Size: {model_size:.2f} MB")

# mAP hesaplama
coco_gt = COCO("chess-pieces-coco/valid/_annotations.coco.json")
coco_dt = coco_gt.loadRes(detections)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
mAP = coco_eval.stats[0]  # mAP @ IoU=0.50:0.95
print(f"mAP: {mAP:.4f}")

# Sonuçları kaydetme
metrics = {
    "confusion_matrix": cm.tolist(),
    "average_inference_time": avg_inference_time,
    "model_size_mb": model_size,
    "mAP": mAP,
}
with open("evaluation_results.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Evaluation results saved.")