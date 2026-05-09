import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from model import SimpleCNN

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Eğitim başlıyor... Cihaz: {device}")
    
    # === PSO SONUÇLARINI BURAYA GİR ===
    BATCH_SIZE = 32        
    LEARNING_RATE = 0.002708  
    BEST_DROPOUT = 0.29     
    # ==================================
    
    PATIENCE = 5 

    train_loader, valid_loader = get_data_loaders(batch_size=BATCH_SIZE)
    if train_loader is None:
        return

    model = SimpleCNN(dropout_rate=BEST_DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    best_v_loss = float('inf')
    patience_counter = 0

    # Grafik için hem Loss hem Accuracy geçmişini tutacak listeler
    history = {
        'train_loss': [], 'valid_loss': [],
        'train_acc': [], 'valid_acc': []
    }

    for epoch in range(30):
        # ==========================================
        # 1. EĞİTİM (TRAIN) AŞAMASI
        # ==========================================
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        for img, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1} [Eğitim]"):
            img, lbl = img.to(device), lbl.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(img)
                loss = criterion(predictions, lbl)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            
            # Doğruluk (Accuracy) Hesaplama
            # Logit değeri 0.0'dan büyükse sınıf 1 (Gerçek), değilse 0 (Sahte)
            predicted_classes = (predictions > 0.0).float()
            train_correct += (predicted_classes == lbl).sum().item()
            train_total += lbl.size(0)
            
        avg_train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total

        # ==========================================
        # 2. DOĞRULAMA (VALIDATION) AŞAMASI
        # ==========================================
        model.eval()
        valid_loss_sum = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for img, lbl in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Test]"):
                img, lbl = img.to(device), lbl.float().unsqueeze(1).to(device)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(img)
                    loss = criterion(predictions, lbl)
                    
                valid_loss_sum += loss.item()
                
                # Doğruluk (Accuracy) Hesaplama
                predicted_classes = (predictions > 0.0).float()
                valid_correct += (predicted_classes == lbl).sum().item()
                valid_total += lbl.size(0)

        avg_v_loss = valid_loss_sum / len(valid_loader)
        valid_acc = valid_correct / valid_total
        
        # Değerleri grafik listelerine ekle
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_v_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        scheduler.step(avg_v_loss) 

        print(f"\nEpoch {epoch+1} Özet -> Kayıp: {avg_train_loss:.4f} (Eğitim), {avg_v_loss:.4f} (Test) | Doğruluk: %{train_acc*100:.2f} (Eğitim), %{valid_acc*100:.2f} (Test)")

        # ==========================================
        # 3. ERKEN DURDURMA VE KAYDETME
        # ==========================================
        if avg_v_loss < best_v_loss:
            best_v_loss = avg_v_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_deepfake_model.pth")
            print(f"🌟 Yeni en iyi model kaydedildi! (Valid Loss: {best_v_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ İyileşme yok. Erken durdurma sayacı: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\n🛑 ERKEN DURDURMA TETİKLENDİ! Eğitim sonlandırılıyor.")
            break 

    # ==========================================
    # 4. İKİLİ GRAFİK ÇİZİMİ (LOSS & ACCURACY)
    # ==========================================
    print("\nGrafikler oluşturuluyor...")
    plt.figure(figsize=(14, 5)) # Yan yana iki grafik için genişliği artırdık
    
    # 1. Grafik: LOSS (Kayıp)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Eğitim Kaybı', color='blue', marker='o')
    plt.plot(history['valid_loss'], label='Doğrulama Kaybı', color='red', marker='s')
    plt.title('Model Öğrenme Eğrisi (Kayıp)', fontsize=12)
    plt.xlabel('Epoch (Tur)', fontsize=10)
    plt.ylabel('Kayıp Değeri (Düşük Olması İyidir)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. Grafik: ACCURACY (Doğruluk)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Eğitim Doğruluğu', color='green', marker='o')
    plt.plot(history['valid_acc'], label='Doğrulama Doğruluğu', color='orange', marker='s')
    plt.title('Model Başarı Eğrisi (Doğruluk)', fontsize=12)
    plt.xlabel('Epoch (Tur)', fontsize=10)
    plt.ylabel('Doğruluk Oranı (Yüksek Olması İyidir)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('egitim_grafikleri.png', dpi=300)
    print("📈 'egitim_grafikleri.png' yüksek kaliteyle kaydedildi. Raporuna eklemeye hazır!")

if __name__ == "__main__":
    train_model()