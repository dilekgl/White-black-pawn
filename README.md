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


- Python 3.8+
- PyTorch
- TensorFlow
- Ultralytics YOLOv8
- OpenCV
- ONNX



## Sonuçlar

Modeller aşağıdaki kriterlere göre karşılaştırılmıştır:
- Ortalama Ortalama Doğruluk (mAP)
- Çıkarım Hızı
- Model Boyutu

Değerlendirme sonuçları `output/` dizininde bulunabilir.

---



