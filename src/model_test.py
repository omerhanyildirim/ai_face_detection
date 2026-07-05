import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from model import SimpleCNN, EfficientNetDeepfake, ResNet18Deepfake, ViTDeepfake
from dataset import get_data_loaders

def test_single_model(model_name, model_class, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'-'*50}")
    print(f"--- {model_name.upper()} TEST EDİLİYOR ---")
    print(f"{'-'*50}")

    if not os.path.exists(weights_path):
        print(f"HATA: {weights_path} bulunamadı! Lütfen önce modeli eğitin.")
        return None

    _, test_loader = get_data_loaders(batch_size=64)
    if test_loader is None: 
        return None

    # Modeli yükle
    model = model_class().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{model_name} İnceleniyor"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda'):
                probs = torch.sigmoid(model(images)).squeeze()
                
                # Batch size 1 gelme ihtimaline karşı boyut düzeltmesi
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)
                    
                preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    target_names = ['FAKE', 'REAL']
    
    # Raporlama
    acc = accuracy_score(all_labels, all_preds) * 100
    print("\n" + "="*40)
    print(f"★ {model_name.upper()} PERFORMANS RAPORU ★")
    print(f"Genel Başarı Oranı: %{acc:.2f}")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Karmaşıklık Matrisini Çiz ve Kaydet
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title(f'{model_name} Karmaşıklık Matrisi')
    plt.tight_layout()
    
    # Görseli kaydet
    save_name = f'test_sonucu_{model_name.lower()}.png'
    plt.savefig(save_name)
    print(f"Hata tablosu '{save_name}' olarak klasöre kaydedildi.")
    
    return acc

def main():
    # Test edilecek 4 modelin tam listesi
    models_to_test = [
        ("ResNet18", ResNet18Deepfake, "best_resnet18_model.pth"),
        ("EfficientNet", EfficientNetDeepfake, "best_effnet_model.pth"),
        ("SimpleCNN", SimpleCNN, "best_deepfake_model.pth"),
        ("ViT", ViTDeepfake, "best_vit_model.pth")
    ]
    
    results = {}
    for name, model_class, path in models_to_test:
        acc = test_single_model(name, model_class, path)
        if acc is not None:
            results[name] = acc
            
    # En Sonda Özet Tablo Çıkar
    print("\n" + "="*50)
    print("TÜM MODELLERİN ÖZET BAŞARI TABLOSU")
    print("="*50)
    for name, acc in results.items():
        print(f"{name:<15}: %{acc:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()