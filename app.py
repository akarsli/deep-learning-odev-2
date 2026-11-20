import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# Sınıf isimleri
CLASSES = [
    'Uçak (Airplane)', 'Otomobil (Automobile)', 'Kuş (Bird)', 'Kedi (Cat)', 
    'Geyik (Deer)', 'Köpek (Dog)', 'Kurbağa (Frog)', 'At (Horse)', 
    'Gemi (Ship)', 'Kamyon (Truck)'
]

# Cihaz ve Model Yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("cifar10_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    if image is None:
        return None, ""
    
    start_time = time.time()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    inf_time = (time.time() - start_time) * 1000
    
    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(10)}
    info_text = f"Tahmin Süresi (Inference Time): {inf_time:.2f} ms"
    
    return confidences, info_text

# Görsel kutusunu sabit yükseklikte tutan ve taşırmayan CSS
custom_css = """
.fixed-image img {
    max-height: 320px !important;
    width: auto !important;
    margin: 0 auto !important;
    object-fit: contain !important;
}
"""

# Gradio Arayüzü
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil", 
        label="Test Fotoğrafı Yükleyin (Telefon Çekimi)", 
        height=320, 
        elem_classes=["fixed-image"]
    ),
    outputs=[
        gr.Label(num_top_classes=3, label="Model Tahmini ve Olasılıklar"),
        gr.Textbox(label="Performans Bilgisi")
    ],
    title="CIFAR-10 Görüntü Sınıflandırma Uygulaması (ResNet18)",
    description="Telefonunuzla çektiğiniz Uçak, Otomobil, Kuş, Kedi, Geyik, Köpek, Kurbağa, At, Gemi veya Kamyon fotoğraflarını yükleyerek modelin tahminini test edin.",
    css=custom_css
)

if __name__ == "__main__":
    demo.launch()