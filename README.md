# ResNet18 Tabanlı Çok Sınıflı Görüntü Sınıflandırma Uygulaması

Bu proje, **Derin Öğrenme** dersi kapsamında açık kaynaklı **CIFAR-10** veri seti üzerinde Transfer Learning (**ResNet18**) yöntemiyle eğitilmiş ve **Gradio** web arayüzü ile canlı kullanıma sunulmuş bir Bilgisayarlı Görü (Computer Vision) uygulamasıdır.

---

## Proje Özeti

| Özellik | Detay |
| :--- | :--- |
| **Problem / Görev** | Çok Sınıflı Görüntü Sınıflandırma (10 Sınıf) |
| **Veri Seti** | CIFAR-10 (40.000 Train, 10.000 Validation, 10.000 Test) |
| **Model Mimarisi** | ResNet18 (ImageNet Ön Eğitimli Ağırlıklar) |
| **Giriş Boyutu** | `224 x 224 x 3` (RGB) |
| **Hiperparametreler** | Learning Rate: `0.001`, Batch Size: `64`, Epochs: `5`, Weight Decay: `1e-4` |
| **Optimizer & Loss** | `AdamW` & `CrossEntropyLoss` |
| **Eğitim Ortamı** | Google Colab (NVIDIA CUDA GPU) |
| **Arayüz (Deployment)** | Gradio Web Uygulaması |

---

## Performans Sonuçları

Google Colab (CUDA GPU) ortamında **1070.51 saniyede** tamamlanan 5 epoch'luk eğitim sonunda 10.000 görüntülük bağımsız test kümesi üzerinde elde edilen metrikler:

* **Test Doğruluğu (Accuracy):** `%89.30`
* **Precision (Macro):** `0.8933`
* **Recall (Macro):** `0.8930`
* **F1-Score (Macro):** `0.8924`
* **Ortalama Tahmin Süresi (Inference Time):** `2.5353 ms / görüntü`

---

## Gerçek Dünya Görüntüleri ile Test Sonuçları

Akıllı telefon kamerası ile çekilen 5 farklı gerçek dünya fotoğrafı Gradio arayüzü üzerinden modele girdi olarak verilmiş ve aşağıdaki sonuçlar elde edilmiştir:

| Görsel No | Gerçek Sınıf | Model Tahmini | Güven Skoru | Tahmin Süresi | Durum |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | Otomobil | Otomobil (Automobile) | `%100` | `19.57 ms` | Başarılı |
| **2** | Kamyon | Kamyon (Truck) | `%97` | `392.58 ms` | Başarılı |
| **3** | Kedi | Kedi (Cat) | `%92` | `41.28 ms` | Başarılı |
| **4** | Uçak | Uçak (Airplane) | `%49` | `19.04 ms` | Başarılı (Düşük Güven) |
| **5** | Gemi | Uçak (Airplane) | `%89` | `28.31 ms` | Başarısız |

---

## Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükleyin
```bash
pip install torch torchvision scikit-learn gradio pillow numpy

```

### 2.Modeli Eğitin

CIFAR-10 veri setini indirir, modeli eğitir, test metriklerini raporlar ve ağırlıkları cifar10_resnet18.pth olarak kaydeder:

```bash
python3 train.py
```

### 3.Web Arayüzünü Başlatın

Eğitilen modeli yükleyerek tarayıcı üzerinden görsel testleri yapabileceğiniz Gradio uygulamasını başlatır:

```bash
python3 app.py
```

### Proje Dosya Yapısı

```bash
├── train.py          # Veri seti hazırlama, ResNet18 Transfer Learning eğitimi ve metrik hesaplama
├── app.py            # Gradio tabanlı etkileşimli görüntü sınıflandırma arayüzü
├── README.md         # Proje dokümantasyonu ve test sonuçları
└── .gitignore        # Veri seti ve geçici sistem dosyaları filtresi
```

