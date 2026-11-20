import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    # 1. Cihaz Ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kullanılan Eğitim Ortamı (Device): {device}")

    # 2. Veri Ön İşleme
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Veri Seti ve Train / Validation / Test Ayrımı (Veri zaten indiği için tekrar indirmeyecek)
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model Mimarisi: Pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # 5. Hiperparametreler, Loss ve Optimizer
    EPOCHS = 5
    LEARNING_RATE = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 6. Eğitim Döngüsü
    print("\nEğitim başlıyor...")
    start_train_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {val_acc:.2f}%")

    total_train_time = time.time() - start_train_time
    print(f"\nToplam Eğitim Süresi: {total_train_time:.2f} saniye")

    # 7. Test Kümesi Performans Değerlendirmesi
    model.eval()
    all_preds = []
    all_labels = []

    start_inf_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    total_inf_time = time.time() - start_inf_time

    avg_inf_time_ms = (total_inf_time / len(test_dataset)) * 1000

    acc = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print("\n--- TEST SONUÇLARI (RAPORA YAZILACAK DEĞERLER) ---")
    print(f"Test Accuracy : %{acc:.2f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"Ortalama Tahmin Süresi (Inference Time): {avg_inf_time_ms:.4f} ms / görüntü")

    # 8. Modeli Kaydetme
    torch.save(model.state_dict(), "cifar10_resnet18.pth")
    print("\nModel 'cifar10_resnet18.pth' olarak kaydedildi.")

if __name__ == '__main__':
    main()