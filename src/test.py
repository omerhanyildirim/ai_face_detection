import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_data_loaders
from model import SimpleCNN, EfficientNetDeepfake, ResNet18Deepfake

def test_ensemble_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test işlemi şu cihazda yapılıyor: {device}")

    _, test_loader = get_data_loaders(batch_size=64)
    if test_loader is None: return

    # 1. MODELLERİ YÜKLE VE AĞIRLIKLARI BELİRLE
    weights = {'res': 0.55, 'eff': 0.30, 'cnn': 0.15}
    models = {}

    print("Karar Komitesi (Ensemble) Modelleri Yükleniyor...")
    
    res_model = ResNet18Deepfake().to(device)
    res_model.load_state_dict(torch.load("best_resnet18_model.pth", map_location=device, weights_only=True))
    res_model.eval()
    models['res'] = res_model

    eff_model = EfficientNetDeepfake().to(device)
    eff_model.load_state_dict(torch.load("best_effnet_model.pth", map_location=device, weights_only=True))
    eff_model.eval()
    models['eff'] = eff_model

    cnn_model = SimpleCNN().to(device)
    cnn_model.load_state_dict(torch.load("best_deepfake_model.pth", map_location=device, weights_only=True))
    cnn_model.eval()
    models['cnn'] = cnn_model

    all_labels = []
    all_preds = []

    print("\n--- 3'lü Karar Komitesi Test Ediliyor ---")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="İnceleniyor"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda'):
                # Her modelden ayrı ayrı olasılık (skor) al
                prob_res = torch.sigmoid(models['res'](images)).squeeze()
                prob_eff = torch.sigmoid(models['eff'](images)).squeeze()
                prob_cnn = torch.sigmoid(models['cnn'](images)).squeeze()

                # Boyut düzeltmesi (Batch size 1 gelme ihtimaline karşı)
                if prob_res.dim() == 0:
                    prob_res = prob_res.unsqueeze(0)
                    prob_eff = prob_eff.unsqueeze(0)
                    prob_cnn = prob_cnn.unsqueeze(0)

                # Formüle göre Ortak Karar Füzyonu: P_nihai = (P_res*0.55) + (P_eff*0.30) + (P_cnn*0.15)
                final_prob = (prob_res * weights['res']) + (prob_eff * weights['eff']) + (prob_cnn * weights['cnn'])
                preds = (final_prob > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\n" + "="*40)
    print("ENŞEMBLE (ÇOKLU MODEL) PERFORMANS RAPORU")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=['FAKE', 'REAL']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title('3\'lü Komite Karmaşıklık Matrisi')
    plt.savefig('test_sonuclari_ensemble.png')
    print("\nHata tablosu 'test_sonuclari_ensemble.png' olarak kaydedildi.")

if __name__ == "__main__":
    test_ensemble_model()