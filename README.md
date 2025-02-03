# Beyaz ve Siyah Piyon Tespiti

Bu proje, `white-pawn` ve `black-pawn` sınıflarını tespit etmek için nesne tespiti modellerini kullanmayı amaçlamaktadır. Projede YOLOv8 ve Fast R-CNN modelleri eğitilip performansları karşılaştırılmıştır.


## Genel Bakış

Projenin amaçları:
- Beyaz ve siyah piyonları tespit etmek için nesne tespiti modellerini eğitmek ve değerlendirmek.
- YOLOv8 ve Fast R-CNN modellerinin performansını karşılaştırmak.
- ONNX modellerini dağıtım için kullanmak.
- ONNX modelini FastAPI uygulamasına yüklemek.

---

## Özellikler

- `white-pawn` ve `black-pawn` sınıflarını tespit etmek için özel nesne tespiti.
- Eğitim ve doğrulama için COCO formatında veri seti desteği.
- YOLOv8 modellerinin ONNX formatına dönüştürülmesi.
- Fast rcnn ve YOLOv8 ile nesne tespiti modellerinin performans karşılaştırması.

---

## Veriseti

Bu projede kullanılan veri seti:
- COCO formatında etiketlenmiş görsellerden oluşur (`train`, `val`, `test` bölümleri).
- YOLOv8 ile kullanmak için:
 ```bash
-python coco_to_yolo.py
```


## Modeller

### YOLOv8
- Ultralytics kütüphanesi kullanılarak eğitildi.
- Dağıtım için ONNX formatına dönüştürüldü.


### Fast R-CNN
- Detaylı performans karşılaştırmaları için kullanıldı.

### FastAPI Uygulaması
FastAPI ile çalışan bir API sunucusunu başlatmak için şu komutları kullanabilirsiniz:
-FastAPI uygulamasını başlatmak için şu komutu kullanın:
```bash
-uvicorn main:app --reload
```
http://127.0.0.1:8000 - Uygulamanın çalıştığı adres
http://127.0.0.1:8000/docs - Swagger arayüzüne erişim

---

## Gereksinimler


- Python 3.8+
- PyTorch
- TensorFlow
- Ultralytics YOLOv8
- OpenCV
- ONNX



## Sonuçlar

### YOLOv8 Modeli 
- mAP: 0.75663
- mAP@50: 0.89377
- mAP@75: 0.99818
- Loss: 0.39614
- Inference Süresi: 4.96766e-05 (çok hızlı)

### Faster R-CNN Modeli
- mAP: 0.7574
- mAP@50: 1.000
- mAP@75: 9629 (Bu değer farklı bir ölçüm formatında olabilir.)
- Loss: 0.2738
- Model Boyutu: 158.08 MB

## Karşılaştırma
Faster R-CNN, doğruluk açısından üstün bir modeldir. Özellikle mAP ve mAP@50 gibi metriklerde daha iyi performans gösterir. Ancak modelin boyutu ve inferans süresi, daha büyük hesaplama kaynakları gerektirebilir.
YOLOv8 ise hızlı inferans süresi ve küçük model boyutu ile gerçek zamanlı uygulamalar için tercih edilen bir seçenek olabilir. Ancak doğruluk açısından Faster R-CNN'e göre biraz daha düşük sonuçlar elde edebilir.

Değerlendirme sonuçları `output/` dizininde bulunabilir.

---



