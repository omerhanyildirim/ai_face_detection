from pso_tune import run_pso
from train import train_model

def main():
    print("="*60)
    print("🤖 TAM OTOMATİK DEEPFAKE EĞİTİM BORU HATTI BAŞLATILIYOR 🤖")
    print("="*60)

    # ---------------------------------------------------------
    # ADIM 1: EFFICIENTNET PSO (ZATEN BULUNDU, SÜREYİ KURTARIYORUZ)
    # ---------------------------------------------------------
    print("\n[ADIM 1/4] EFFICIENTNET PSO ZATEN HESAPLANDI (ATLANARAK DEVAM EDİLİYOR...)")
    # Terminalde gördüğümüz senin sonuçlarını buraya manuel yazdık
    best_lr_eff = 0.00041
    best_drop_eff = 0.30
    best_batch_eff = 32

    # ---------------------------------------------------------
    # ADIM 2: SIMPLE CNN PSO HESAPLAMASI
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("[ADIM 2/4] SIMPLE CNN İÇİN PSO HESAPLAMASI BAŞLIYOR...")
    best_lr_cnn, best_drop_cnn, best_batch_cnn = run_pso(model_name="SimpleCNN")

    # ---------------------------------------------------------
    # ADIM 3: EFFICIENTNET TRAIN
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("[ADIM 3/4] EFFICIENTNET EĞİTİMİ (TRAIN) BAŞLIYOR...")
    train_model(
        model_name="EfficientNet", 
        learning_rate=best_lr_eff, 
        dropout_rate=best_drop_eff, 
        batch_size=best_batch_eff
    )

    # ---------------------------------------------------------
    # ADIM 4: SIMPLE CNN TRAIN
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("[ADIM 4/4] SIMPLE CNN EĞİTİMİ (TRAIN) BAŞLIYOR...")
    train_model(
        model_name="SimpleCNN", 
        learning_rate=best_lr_cnn, 
        dropout_rate=best_drop_cnn, 
        batch_size=best_batch_cnn
    )

    print("\n" + "="*60)
    print("🎉 TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print("Elinde iki adet eğitilmiş modelin ve grafiklerin var.")
    print("Artık predict.py dosyasını çalıştırıp test edebilirsin.")
    print("="*60)

if __name__ == "__main__":
    main()