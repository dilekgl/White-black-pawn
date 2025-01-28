# Beyaz ve Siyah Piyon Tespiti

Bu proje, `white-pawn` ve `black-pawn` sınıflarını tespit etmek için nesne tespiti modellerini kullanmayı amaçlamaktadır. Projede YOLOv8 ve Fast R-CNN modelleri eğitilip performansları karşılaştırılmıştır.


## Genel Bakış

Projenin amaçları:
- Beyaz ve siyah piyonları tespit etmek için nesne tespiti modellerini eğitmek ve değerlendirmek.
- YOLOv8 ve Fast R-CNN modellerinin performansını karşılaştırmak.
- ONNX modellerini dağıtım için kullanmak.

---

## Özellikler

- `white-pawn` ve `black-pawn` sınıflarını tespit etmek için özel nesne tespiti.
- Eğitim ve doğrulama için COCO formatında veri seti desteği.
- YOLOv8 modellerinin ONNX formatına dönüştürülmesi.
- Farklı nesne tespiti modellerinin performans karşılaştırması.

---

## Veriseti

Bu projede kullanılan veri seti:
- COCO formatında etiketlenmiş görsellerden oluşur (`train`, `val`, `test` bölümleri).
- `white-pawn` ve `black-pawn` sınıflarına ait sınırlayıcı kutuları içerir.


## Modeller

### YOLOv8
- Ultralytics kütüphanesi kullanılarak eğitildi.
- Dağıtım için ONNX formatına dönüştürüldü.


### Fast R-CNN
- Detaylı performans karşılaştırmaları için kullanıldı.


---

## Gereksinimler

Aşağıdaki yazılımların yüklü olduğundan emin olun:

- Python 3.8+
- PyTorch
- TensorFlow
- Ultralytics YOLOv8
- OpenCV
- ONNX


## Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/dilekgl/White-black-pawn.git
   cd White-black-pawn
   ```

2. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Veri setini hazırlayın:
   - Veri seti dosyalarını `data/` dizinine yerleştirin.
   - COCO formatındaki JSON dosyalarının mevcut olduğundan emin olun.

---

## Kullanım

### YOLOv8 Eğitimi

YOLOv8'i eğitmek için:
```bash
python train.py --model yolov8n.yaml --data data.yaml --epochs 50
```

YOLOv8 modelini ONNX'e dönüştürmek için:
```bash
python export.py --weights best.pt --img-size 640 --format onnx
```

### Fast R-CNN Eğitimi

Fast R-CNN modelini eğitmek için:
```bash
python train_fastrcnn.py --data data.yaml --epochs 50
```

Modelin performansını değerlendirmek için:
```bash
python evaluate_fastrcnn.py --model output/model.pth --data data.yaml
```

---

## Sonuçlar

Modeller aşağıdaki kriterlere göre karşılaştırılmıştır:
- Ortalama Ortalama Doğruluk (mAP)
- Çıkarım Hızı
- Model Boyutu

Değerlendirme sonuçları `results/` dizininde bulunabilir.

---

## Gelecek Çalışmalar

- Daha hızlı çıkarım için modelleri optimize etme.
- Daha çeşitli görsellerle veri setini genişletme.
- En iyi performans gösteren modeli gerçek zamanlı bir uygulama olarak dağıtma.

---

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.
