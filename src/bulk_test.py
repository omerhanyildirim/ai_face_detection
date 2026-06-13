import torch
from tqdm import tqdm
from dataset import get_data_loaders
from model import SimpleCNN, EfficientNetDeepfake, ResNet18Deepfake

def test_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test işlemi başlatılıyor. Cihaz: {device}")

    # Veri setini DataLoader ile hızlıca yükle
    _, test_loader = get_data_loaders(batch_size=64)
    if test_loader is None: 
        print("Test verisi bulunamadı!")
        return

    print("Modeller Yüklenecek...")
    models = {}
    
    # 1. SimpleCNN Yükle
    try:
        cnn = SimpleCNN().to(device)
        cnn.load_state_dict(torch.load("best_deepfake_model.pth", map_location=device, weights_only=True))
        cnn.eval()
        models['cnn'] = cnn
    except: print("Uyarı: SimpleCNN (best_deepfake_model.pth) bulunamadı.")

    # 2. EfficientNet Yükle
    try:
        eff = EfficientNetDeepfake().to(device)
        eff.load_state_dict(torch.load("best_effnet_model.pth", map_location=device, weights_only=True))
        eff.eval()
        models['eff'] = eff
    except: print("Uyarı: EfficientNet (best_effnet_model.pth) bulunamadı.")

    # 3. ResNet18 Yükle
    try:
        res = ResNet18Deepfake().to(device)
        res.load_state_dict(torch.load("best_resnet18_model.pth", map_location=device, weights_only=True))
        res.eval()
        models['res'] = res
    except: print("Uyarı: ResNet18 (best_resnet18_model.pth) bulunamadı.")

    if not models:
        print("Test edilecek hiçbir model bulunamadı! Lütfen önce eğitim yapın.")
        return

    # İstatistik tutucular
    correct_preds = {'cnn': 0, 'eff': 0, 'res': 0}
    total_samples = 0

    print("\n--- Tüm Modeller Test Ediliyor ---")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Görseller İnceleniyor"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda'):
                # SimpleCNN Tahmini
                if 'cnn' in models:
                    prob_cnn = torch.sigmoid(models['cnn'](images)).squeeze()
                    if prob_cnn.dim() == 0: prob_cnn = prob_cnn.unsqueeze(0)
                    preds_cnn = (prob_cnn > 0.5).float()
                    correct_preds['cnn'] += (preds_cnn == labels).sum().item()

                # EfficientNet Tahmini
                if 'eff' in models:
                    prob_eff = torch.sigmoid(models['eff'](images)).squeeze()
                    if prob_eff.dim() == 0: prob_eff = prob_eff.unsqueeze(0)
                    preds_eff = (prob_eff > 0.5).float()
                    correct_preds['eff'] += (preds_eff == labels).sum().item()

                # ResNet18 Tahmini
                if 'res' in models:
                    prob_res = torch.sigmoid(models['res'](images)).squeeze()
                    if prob_res.dim() == 0: prob_res = prob_res.unsqueeze(0)
                    preds_res = (prob_res > 0.5).float()
                    correct_preds['res'] += (preds_res == labels).sum().item()

            total_samples += labels.size(0)

    # Doğruluk (Accuracy) Hesaplamaları
    acc_cnn = (correct_preds['cnn'] / total_samples) * 100 if 'cnn' in models else 0
    acc_eff = (correct_preds['eff'] / total_samples) * 100 if 'eff' in models else 0
    acc_res = (correct_preds['res'] / total_samples) * 100 if 'res' in models else 0

    # İstediğin Formatta Sonuç Ekranı
    print("\n" + "="*50)
    print("MODELLERİN TEST VERİ SETİ BAŞARI ORANLARI")
    print("="*50)
    if 'cnn' in models: print(f"SimpleCNN başarı oranı    : %{acc_cnn:.2f}")
    if 'eff' in models: print(f"EfficientNet başarı oranı : %{acc_eff:.2f}")
    if 'res' in models: print(f"ResNet18 başarı oranı     : %{acc_res:.2f}")
    print("="*50)
    print(f"Toplam test edilen görsel sayısı: {total_samples}")

if __name__ == "__main__":
    test_all_models()