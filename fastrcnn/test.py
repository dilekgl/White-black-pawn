import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Kullanıcı tanımlı veri kümesi sınıfı
class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Görüntüyü yükleme
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Veriyi normalize et ve tensöre çevir
        if self.transform:
            image = self.transform(image)

        # COCO formatında anotasyonları hazırla
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        return image, target

# Modeli yükleme fonksiyonu
def load_model(model_path):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# mAP hesaplama fonksiyonu
def evaluate(model, data_loader, device):
    model.to(device)
    coco_gt = data_loader.dataset.coco
    coco_results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                img_id = int(target['image_id'].item())
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    coco_results.append({
                        'image_id': img_id,
                        'category_id': label,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': float(score)
                    })

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # AP @ IoU=0.50:0.95

# Ana çalışma kodu
if __name__ == "__main__":
    # Kullanıcı ayarları
    model_path = "best_model.pth"
    image_dir = "fastrcnn/chess-pieces-fastrcnn/test/images"
    annotation_path = "chess-pieces-coco/test/_annotations.coco.json"

    # Görüntü dönüşümleri
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Veri kümesini ve yükleyiciyi oluştur
    dataset = CustomDataset(image_dir, annotation_path, transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Modeli yükle ve değerlendir
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    print("Evaluating model...")
    mAP = evaluate(model, data_loader, device)
    print(f"Model mAP: {mAP:.4f}")