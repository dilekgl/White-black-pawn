import torch
import torchvision
from torchvision import transforms
import time
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def main():
    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load('best_model.pth'), strict=False)
    model.eval()
    model = model.to('cuda')

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor()
    ])

    # Calculate model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Size: {model_size / 1e6:.2f} MB')

    # Inference time calculation
    def compute_inference_time(model, data_loader):
        start_time = time.time()
        with torch.no_grad():
            for images, _ in data_loader:
                images = [image.to('cuda') for image in images]
                model(images)
        end_time = time.time()
        inference_time = (end_time - start_time) / len(data_loader.dataset)
        return inference_time

    # mAP calculation
    def compute_map(model, data_loader, iou_threshold=0.1):
        model.eval()
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for images, targets in data_loader:
                images = [image.to('cuda') for image in images]
                outputs = model(images)

                for i, output in enumerate(outputs):
                    pred_boxes = output['boxes'].cpu()
                    pred_labels = output['labels'].cpu()

                    gt_boxes = torch.tensor([t['bbox'] for t in targets[i]], dtype=torch.float32)

                    # Ensure gt_boxes is not empty and reshape if necessary
                    if gt_boxes.numel() > 0:
                        gt_boxes = gt_boxes.view(-1, 4)  # Ensure correct shape
                        gt_boxes[:, 2:] += gt_boxes[:, :2]  # Convert to [x_min, y_min, x_max, y_max]
                    else:
                        gt_boxes = torch.empty((0, 4), dtype=torch.float32)

                    if len(pred_boxes) == 0:
                        fn += len(gt_boxes)
                        continue

                    ious = box_iou(pred_boxes, gt_boxes)
                    matched = ious.max(dim=1).values >= iou_threshold

                    tp += matched.sum().item()
                    fp += len(pred_boxes) - matched.sum().item()
                    fn += len(gt_boxes) - matched.sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        ap = (precision * recall) / (precision + recall + 1e-10)
        for i, output in enumerate(outputs):
            print(f"Image {i} - Predicted Boxes: {output['boxes'].cpu().numpy()}")
            print(f"Image {i} - Confidence Scores: {output['scores'].cpu().numpy()}")
            print(f"Image {i} - Predicted Labels: {output['labels'].cpu().numpy()}")

        print(f"Predicted Labels: {pred_labels}")
        
        return ap

    # Load COCO test dataset
    coco_test_data = CocoDetection(root='fastrcnn/chess-pieces-fastrcnn/test/images',
                                   annFile='chess-pieces-coco/test/_annotations.coco.json',
                                   transform=transform)

    test_loader = DataLoader(coco_test_data, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Calculate inference time
    inference_time = compute_inference_time(model, test_loader)
    print(f'Inference Time: {inference_time:.4f} seconds per image')

    # Calculate mAP
    map_score = compute_map(model, test_loader)
    print(f'mAP: {map_score:.4f}')

if __name__ == '__main__':
    main()

