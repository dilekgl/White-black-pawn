import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import numpy as np
import matplotlib
from torchvision.models.detection import fasterrcnn_resnet50_fpn
matplotlib.use('Agg')  # Non-interactive backend kullan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Define transformations with normalization
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image, target

# Dataset function
def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )

# Load datasets
train_dataset = get_coco_dataset("fastrcnn/chess-pieces-fastrcnn/train/images", "chess-pieces-coco/train/_annotations.coco.json")
val_dataset = get_coco_dataset("fastrcnn/chess-pieces-fastrcnn/valid/images", "chess-pieces-coco/valid/_annotations.coco.json")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 2  # Background + white-pawn + black-pawn
model = get_model(num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Load the best saved model
model = fasterrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load('best_model.pth'), strict=False)
# Function to visualize a batch of images with bounding boxes
def visualize_batch(images, targets, save_dir="fastrcnn/fastrcnn-pawn-model/output/batches"):
    os.makedirs(save_dir, exist_ok=True)
    for idx, (img, target_list) in enumerate(zip(images, targets)):  # target_list, her bir görüntü için nesne listesi
        img = img.permute(1, 2, 0).cpu().numpy()  # Tensor'ı numpy'a çevir ve channel'ları düzenle
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # Normalizasyonu geri al
        img = np.clip(img, 0, 1)  # Değerleri [0, 1] aralığına sıkıştır

        plt.figure()
        plt.imshow(img)
        
        # Her bir nesne için bounding box çiz
        for target in target_list:  # target_list, her bir nesne için bir sözlük içerir
            box = target['bbox']  # COCO formatı: [x_min, y_min, width, height]
            label = target['category_id']  # Nesnenin ID'si
            
            # COCO formatını [x1, y1, x2, y2] formatına çevir
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height
            
            # Bounding box çiz
            plt.gca().add_patch(plt.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none'))
            
            # Nesnenin ID'sini yaz
            plt.text(x1, y1, f'ID: {label}', color='white', backgroundcolor='red', fontsize=8)
        
        plt.title(f"Image {idx}")
        plt.savefig(f"{save_dir}/batch_{idx}.png")
        plt.close()

# Evaluate the model and generate confusion matrix, PR curve, and ROC curve
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_labels = output["labels"].cpu().numpy().tolist()
                true_labels = [obj["category_id"] for obj in targets[i]]
                scores = output["scores"].cpu().numpy().tolist()

                # Filtreleme: Sadece geçerli sınıfları (5 ve 11) kabul et
                filtered_true_labels = [label for label in true_labels if label in VALID_CLASSES]
                filtered_pred_labels = [label for label in pred_labels if label in VALID_CLASSES.values()]

                # Eşleşen sayıda örnek al
                min_length = min(len(filtered_true_labels), len(filtered_pred_labels))
                if min_length == 0:
                    continue  # Eğer hiç geçerli örnek yoksa atla

                filtered_true_labels = filtered_true_labels[:min_length]
                filtered_pred_labels = filtered_pred_labels[:min_length]

                all_labels.extend(filtered_true_labels)
                all_preds.extend(filtered_pred_labels)

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", conf_matrix)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['black-pawn', 'white-pawn'], yticklabels=['black-pawn', 'white-pawn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("fastrcnn/fastrcnn-pawn-model/output/confusion_matrix.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds, pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig("fastrcnn/fastrcnn-pawn-model/output/pr_curve.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("fastrcnn/fastrcnn-pawn-model/output/roc_curve.png")
    plt.close()

    # Save metrics to CSV
    metrics = {
        'Confusion Matrix': [conf_matrix.tolist()],  # Listeye çevir
        'ROC AUC': [float(roc_auc)],  # Tek bir float değerini liste içine al
        'Precision-Recall Curve': [list(zip(precision.tolist(), recall.tolist()))],  # Listeye çevir
        'ROC Curve': [list(zip(fpr.tolist(), tpr.tolist()))]  # Listeye çevir
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.to_csv("fastrcnn/fastrcnn-pawn-model/output/metrics.csv")

    return conf_matrix, precision, recall, fpr, tpr, roc_auc

# Create output directory
os.makedirs("fastrcnn/fastrcnn-pawn-model/output", exist_ok=True)

# Visualize a training batch
images, targets = next(iter(train_loader))
visualize_batch(images, targets)

# Evaluate the model
VALID_CLASSES = {5: 0, 11: 1}  # Geçerli sınıflar
conf_matrix, precision, recall, fpr, tpr, roc_auc = evaluate_model(model, val_loader, device)

print("Evaluation completed. Outputs saved in the 'output' directory.")