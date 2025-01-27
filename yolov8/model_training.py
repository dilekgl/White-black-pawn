from ultralytics import YOLO
"""
# Model oluştur
model=YOLO("yolov8n.pt")
model = YOLO("yolov8/runs/detect/yolo_pawn_model/weights/best.pt")
results = model.predict(source="./chess-pieces-yolo/valid/images", save=True)
"""
"""
# pt-->onnx
model.export(format="onnx", imgsz=[640,640], opset=17)
#print("Model başarıyla ONNX formatına dönüştürüldü!")
"""
"""
# Modeli eğit
result= model.train(
            data="./data.yaml",      # YAML dosyasının yolu
            epochs=50,               # Eğitim dönemi sayısı
            imgsz=640,               # Görüntü boyutu
            batch=16,                # Batch boyutu
            name="yolo_pawn_model"   # Model adını belirleme
        )

# Modeli değerlendirme (isteğe bağlı)
metrics = model.val()
print(metrics)
"""