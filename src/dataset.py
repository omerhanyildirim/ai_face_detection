#Eğitim tamamlandı! Model 'deepfake_cnn_model.pth' olarak ana dizine kaydedildi. -> 0.9851

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir="../data/real_vs_fake", batch_size=64, image_size=128):
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # ==========================================
    # 1. EĞİTİM (TRAIN) İÇİN ZORLAŞTIRILMIŞ DÖNÜŞÜMLER
    # ==========================================
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        
        # --- DATA AUGMENTATION (EZBER BOZUCULAR) ---
        transforms.RandomHorizontalFlip(p=0.5), # %50 ihtimalle aynalar (Asimetri ezberini bozar)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Işık/Renk ezberini bozar
        transforms.RandomRotation(degrees=10), # Kafanın duruş açısını hafif değiştirir
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # EN KRİTİK OLAN! Piksel izlerini siler
        # ------------------------------------------
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # ==========================================
    # 2. TEST İÇİN TEMİZ DÖNÜŞÜMLER (Asla dokunulmaz)
    # ==========================================
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # Veri setlerini yüklerken farklı transformları veriyoruz
    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
        
        print(f"Eğitim (Train) verisi yüklendi: {len(train_dataset)} görüntü. (Augmentation Aktif)")
        print(f"Test verisi yüklendi: {len(test_dataset)} görüntü. (Temiz)")
        
    except Exception as e:
        print("HATA: Veri seti okunamadı! Klasör yollarını kontrol et.")
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == "__main__":
    print("Veri yükleyici test ediliyor...")
    train_loader, test_loader = get_data_loaders()