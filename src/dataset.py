#Eğitim tamamlandı! Model 'deepfake_cnn_model.pth' olarak ana dizine kaydedildi. -> 0.9851

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir="../data/real_vs_fake", batch_size=64, image_size=128):
    """
    Veri setini 'train' ve 'test' klasörlerinden okur, ön işlemlerden geçirir.
    """
    
    # 1. Klasör yollarını belirle
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 2. Görüntülere uygulanacak ön işlemler (Transforms)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # 3. Veri setlerini ImageFolder ile doğrudan klasörlerden yükle
    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        
        print(f"Eğitim (Train) verisi yüklendi: {len(train_dataset)} görüntü.")
        print(f"Test verisi yüklendi: {len(test_dataset)} görüntü.")
        print(f"Sınıflar: {train_dataset.class_to_idx}")
    except Exception as e:
        print("HATA: Veri seti okunamadı! Klasör yollarını kontrol et.")
        print(f"Detay: {e}")
        return None, None

    # 4. DataLoader oluşturma (Ekran kartına batch'ler halinde göndermek için)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == "__main__":
    print("Veri yükleyici test ediliyor...")
    train_loader, test_loader = get_data_loaders()