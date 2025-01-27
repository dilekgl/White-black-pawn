import json
import os
from pathlib import Path



# Giriş ve çıkış yolları
input_file = "./chess-pieces-coco/test/_annotations.coco.json"
output_dir = "./chess-pieces-yolo/test/labels/"

# Çıkış klasörünü oluştur
os.makedirs(output_dir, exist_ok=True)

# JSON'u yükle
with open(input_file, "r") as f:
    data = json.load(f)
target_categories = {"white-pawn": 0, "black-pawn": 1}

# Sadece hedef kategoriler için eşleme oluştur
categories = {
    cat["id"]: target_categories[cat["name"]]
    for cat in data["categories"]
    if cat["name"] in target_categories
}
# Görüntü ve anotasyonları işleme
for image in data["images"]:
    image_id = image["id"]
    image_height = image["height"]
    image_width = image["width"]
    image_name = Path(image["file_name"]).stem

    # İlgili anotasyonları filtrele
    annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] == image_id and ann["category_id"] in categories
    ]

    # Eğer geçerli anotasyon yoksa, boş dosya oluşturma
    if not annotations:
        print(f"Skipping {image_name}: No valid annotations found.")
        continue

    # Çıkış dosyasını aç
    with open(os.path.join(output_dir, f"{image_name}.txt"), "w") as txt_file:
        for ann in annotations:
            category_id = ann["category_id"]
            bbox = ann["bbox"]  # COCO formatında: [x_min, y_min, width, height]

            # Kategori ID'yi YOLO sınıf ID'ye çevir
            yolo_class_id = categories[category_id]

            # Bounding box dönüşümü (COCO -> YOLO)
            x_min, y_min, box_width, box_height = bbox
            x_center = x_min + box_width / 2
            y_center = y_min + box_height / 2

            # Normalize et
            x_center /= image_width
            y_center /= image_height
            box_width /= image_width
            box_height /= image_height

            # YOLO formatında yaz
            txt_file.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print(f"YOLO formatına dönüşüm tamamlandı. Çıkış dosyaları: {output_dir}")