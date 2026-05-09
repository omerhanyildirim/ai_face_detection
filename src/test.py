import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_data_loaders
from model import SimpleCNN

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test işlemi şu cihazda yapılıyor: {device}")

    # 1. Veriyi Yükle (Eğitimde ayırdığımız test setini kullanıyoruz)
    _, test_loader = get_data_loaders(batch_size=64)

    # 2. Modeli Yükle
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("best_deepfake_model.pth", map_location=device, weights_only=True))
    model.eval()

    all_labels = []
    all_preds = []

    print("\n--- Veri Seti Üzerinde Tespit Başlıyor ---")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="İnceleniyor"):
            images = images.to(device)
            labels = labels.to(device)

            # Tahmin yap (Logit olarak gelir)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                # Olasılığa çevir ve 0.5 sınırına göre sınıflandır
                preds = (torch.sigmoid(outputs) > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 3. RAPORLAMA
    print("\n" + "="*30)
    print("DETAYLI PERFORMANS RAPORU")
    print("="*30)
    # Precision, Recall ve F1-Score gibi akademik metrikler
    print(classification_report(all_labels, all_preds, target_names=['REAL', 'FAKE']))

    # 4. Karmaşıklık Matrisi (Confusion Matrix) Çizimi
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title('Karmaşıklık Matrisi (Hata Tablosu)')
    plt.savefig('test_sonuclari_matris.png')
    print("\n📈 Hata tablosu 'test_sonuclari_matris.png' olarak kaydedildi.")

if __name__ == "__main__":
    test_model()