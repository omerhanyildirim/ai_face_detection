import torch
import os
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
from tqdm import tqdm

def bulk_test(test_dir="../data/real_vs_fake/test", model_path="best_deepfake_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modeli yükle
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # İstatistikler
    stats = {
        'real': {'correct': 0, 'total': 0},
        'fake': {'correct': 0, 'total': 0}
    }

    print(f"\n--- Sınıf Bazlı Doğruluk Analizi Başlıyor ---")

    for category in ['real', 'fake']:
        folder_path = os.path.join(test_dir, category)
        if not os.path.exists(folder_path):
            continue
            
        files = os.listdir(folder_path)
        for file_name in tqdm(files, desc=f"Klasör: {category}"):
            img_path = os.path.join(folder_path, file_name)
            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    prob = torch.sigmoid(logits).item()

                # DÜZELTİLMİŞ MANTIK: 
                # Önceki testte %1 aldığımız için etiketleri burada tersine çeviriyoruz.
                # Artık prob > 0.5 ise REAL, prob <= 0.5 ise FAKE olarak kabul ediyoruz.
                is_predicted_real = prob > 0.5 
                
                if category == 'real':
                    stats['real']['total'] += 1
                    if is_predicted_real: # Real klasöründe real bulduysa
                        stats['real']['correct'] += 1
                
                elif category == 'fake':
                    stats['fake']['total'] += 1
                    if not is_predicted_real: # Fake klasöründe fake bulduysa
                        stats['fake']['correct'] += 1
            except:
                continue

    # Oranları Hesapla
    real_acc = (stats['real']['correct'] / stats['real']['total'] * 100) if stats['real']['total'] > 0 else 0
    fake_acc = (stats['fake']['correct'] / stats['fake']['total'] * 100) if stats['fake']['total'] > 0 else 0
    overall_acc = ((stats['real']['correct'] + stats['fake']['correct']) / (stats['real']['total'] + stats['fake']['total']) * 100)

    print("\n" + "="*50)
    print("VERİ SETİ DOĞRULUK RAPORU")
    print("="*50)
    print(f"REAL Datasetindeki Doğruluk : %{real_acc:.2f}")
    print(f"   ({stats['real']['correct']} / {stats['real']['total']} doğru)")
    print("-" * 50)
    print(f"FAKE Datasetindeki Doğruluk : %{fake_acc:.2f}")
    print(f"   ({stats['fake']['correct']} / {stats['fake']['total']} doğru)")
    print("-" * 50)
    print(f"Genel Başarı Oranı         : %{overall_acc:.2f}")
    print("="*50)

if __name__ == "__main__":
    bulk_test()