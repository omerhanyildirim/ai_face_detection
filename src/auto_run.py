from pso_tune import run_pso
from train import train_model

def main():
    print("="*60)
    print("🤖 VISION TRANSFORMER (ViT) EĞİTİM BORU HATTI BAŞLATILIYOR 🤖")
    print("="*60)
    print("Not: EfficientNet, SimpleCNN ve ResNet18 zaten eğitildiği için güvende.\n")

    # ---------------------------------------------------------
    # SADECE ViT İÇİN SÜRECİ ÇALIŞTIRIYORUZ
    # ---------------------------------------------------------
    
    # 1. ViT İÇİN PSO
    print("[ADIM 1/2] VISION TRANSFORMER (ViT) İÇİN PSO HESAPLAMASI BAŞLIYOR...")
    best_lr_vit, best_drop_vit, best_batch_vit = run_pso(model_name="ViT")

    # 2. ViT İÇİN EĞİTİM
    print("\n" + "="*60)
    print("[ADIM 2/2] VISION TRANSFORMER (ViT) EĞİTİMİ (TRAIN) BAŞLIYOR...")
    train_model(
        model_name="ViT", 
        learning_rate=best_lr_vit, 
        dropout_rate=best_drop_vit, 
        batch_size=best_batch_vit
    )

    print("\n" + "="*60)
    print("🎉 VISION TRANSFORMER (ViT) BAŞARIYLA SİSTEME EKLENDİ VE EĞİTİLDİ!")
    print("Artık 4'lü Akademik Komite hazır. predict.py dosyasını çalıştırıp test edebilirsin.")
    print("="*60)

    # =====================================================================
    # NOT: İLERİDE TÜM MODELLERİ SIFIRDAN EĞİTMEK İSTERSEN AŞAĞIDAKİ
    # YORUM SATIRLARINI KALDIRABİLİRSİN. (ŞU ANKİ EMEKLERİNİ KORUMAK İÇİN KAPALI)
    # =====================================================================
    """
    # EFFICIENTNET
    best_lr_eff, best_drop_eff, best_batch_eff = run_pso("EfficientNet")
    train_model("EfficientNet", best_lr_eff, best_drop_eff, best_batch_eff)
    
    # RESNET18
    best_lr_res, best_drop_res, best_batch_res = run_pso("ResNet18")
    train_model("ResNet18", best_lr_res, best_drop_res, best_batch_res)
    
    # SIMPLECNN
    best_lr_cnn, best_drop_cnn, best_batch_cnn = run_pso("SimpleCNN")
    train_model("SimpleCNN", best_lr_cnn, best_drop_cnn, best_batch_cnn)
    """

if __name__ == "__main__":
    main()