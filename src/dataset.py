import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing

def get_data_loaders(data_dir="../data/real_vs_fake", batch_size=64, image_size=128):
    train_dir = os.path.join(data_dir, 'train')

    valid_dir = os.path.join(data_dir, 'valid')
    if not os.path.exists(valid_dir):
        valid_dir = os.path.join(data_dir, 'test')

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=5), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)
    except Exception as e:
        print("\nHATA: Veri seti okunamadı!")
        print(f"Aranan Train Klasörü: {os.path.abspath(train_dir)}")
        print(f"Aranan Test/Valid Klasörü: {os.path.abspath(valid_dir)}")
        print(f"Sistem Mesajı: {e}\n")
        return None, None

    num_workers = min(4, multiprocessing.cpu_count())

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,                                    
        persistent_workers=True if num_workers > 0 else False
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,                                   
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, valid_loader