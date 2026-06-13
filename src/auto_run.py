from train import train_model

def main():
    print("="*60)
    print("DEEPFAKE TESPİT SİSTEMİ - EĞİTİM YÖNETİCİSİ")
    print("="*60)
    print("Not: EfficientNet, ResNet18 ve SimpleCNN modelleri mevcutta eğitilmiştir.")
    print("Modelleri baştan eğitmek isterseniz aşağıdaki yorum satırlarını kaldırın.\n")

    """
    # 1. ResNet18 Eğitimi
    train_model("ResNet18", learning_rate=0.0001, dropout_rate=0.3, batch_size=64)
    
    # 2. EfficientNet Eğitimi
    train_model("EfficientNet", learning_rate=0.0001, dropout_rate=0.3, batch_size=64)
    
    # 3. SimpleCNN Eğitimi
    train_model("SimpleCNN", learning_rate=0.001, dropout_rate=0.5, batch_size=64)
    """

    print("Tüm modeller savaşa hazır! Lütfen arayüz için predict.py'yi çalıştırın.")

if __name__ == "__main__":
    main()