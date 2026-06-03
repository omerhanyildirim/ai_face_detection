import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from model import SimpleCNN, EfficientNetDeepfake, ResNet18Deepfake, ViTDeepfake

def train_model(model_name, learning_rate, dropout_rate, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 ASIL EĞİTİM BAŞLIYOR: {model_name} | Cihaz: {device}")
    
    PATIENCE = 5 

    train_loader, valid_loader = get_data_loaders(batch_size=batch_size)
    if train_loader is None: return

    # OTOMATİK MODEL SEÇİMİ VE KAYIT YOLU (4 MODEL DESTEKLİ)
    if model_name == "SimpleCNN":
        model = SimpleCNN(dropout_rate=dropout_rate).to(device)
        save_path = "best_deepfake_model.pth"
    elif model_name == "EfficientNet":
        model = EfficientNetDeepfake(dropout_rate=dropout_rate).to(device)
        save_path = "best_effnet_model.pth"
    elif model_name == "ResNet18":
        model = ResNet18Deepfake(dropout_rate=dropout_rate).to(device)
        save_path = "best_resnet18_model.pth"
    elif model_name == "ViT":
        model = ViTDeepfake(dropout_rate=dropout_rate).to(device)
        save_path = "best_vit_model.pth"

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    best_v_loss = float('inf')
    patience_counter = 0

    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}

    for epoch in range(30):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        
        train_loop = tqdm(train_loader, desc=f"  [Train - {model_name} Epoch {epoch+1}]")
        for img, lbl in train_loop:
            img, lbl = img.to(device), lbl.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(img)
                loss = criterion(predictions, lbl)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            predicted_classes = (predictions > 0.0).float()
            train_correct += (predicted_classes == lbl).sum().item()
            train_total += lbl.size(0)
            
        avg_train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        valid_loss_sum, valid_correct, valid_total = 0.0, 0, 0
        
        valid_loop = tqdm(valid_loader, desc=f"  [Test  - {model_name} Epoch {epoch+1}]")
        with torch.no_grad():
            for img, lbl in valid_loop:
                img, lbl = img.to(device), lbl.float().unsqueeze(1).to(device)
                with torch.amp.autocast('cuda'):
                    predictions = model(img)
                    loss = criterion(predictions, lbl)
                valid_loss_sum += loss.item()
                predicted_classes = (predictions > 0.0).float()
                valid_correct += (predicted_classes == lbl).sum().item()
                valid_total += lbl.size(0)

        avg_v_loss = valid_loss_sum / len(valid_loader)
        valid_acc = valid_correct / valid_total
        
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_v_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        scheduler.step(avg_v_loss) 

        print(f"Epoch {epoch+1} -> Kayıp: {avg_train_loss:.4f} (Eğitim), {avg_v_loss:.4f} (Test) | Doğruluk: %{train_acc*100:.2f} (Eğitim), %{valid_acc*100:.2f} (Test)")

        if avg_v_loss < best_v_loss:
            best_v_loss = avg_v_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Yeni en iyi {model_name} modeli kaydedildi! (Valid Loss: {best_v_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("🛑 Erken Durdurma Tetiklendi!")
            break 

    # Grafikleri modele özel kaydet
    plt.figure(figsize=(14, 5)) 
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Eğitim', color='blue')
    plt.plot(history['valid_loss'], label='Test', color='red')
    plt.title(f'{model_name} - Kayıp (Loss)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Eğitim', color='green')
    plt.plot(history['valid_acc'], label='Test', color='orange')
    plt.title(f'{model_name} - Doğruluk (Accuracy)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'egitim_grafikleri_{model_name}.png', dpi=300)
    print(f"📈 Grafik kaydedildi: egitim_grafikleri_{model_name}.png")