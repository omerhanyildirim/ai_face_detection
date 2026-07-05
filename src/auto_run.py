from pso_tune import run_pso

def main():
    print("="*60)
    print("DEEPFAKE TESPİT SİSTEMİ - SADECE PSO HESAPLAYICI")
    print("="*60)
    print("Asıl eğitim YASAKLANDI. Yalnızca 5 parçacık ve 4 iterasyonluk PSO optimizasyonu çalıştırılıyor...\n")

    modeller = ["ViT"]
    pso_sonuclari = {}

    for model_adi in modeller:
        best_lr, best_drop, best_batch = run_pso(model_adi)
        pso_sonuclari[model_adi] = {
            "LR": best_lr,
            "Dropout": best_drop,
            "Batch": best_batch
        }

    print("\n\n" + "#"*60)
    print("NİHAİ PSO OPTİMİZASYON RAPORU".center(60))
    print("#"*60)
    for model_adi, sonuclar in pso_sonuclari.items():
        print(f"Model: {model_adi:<15} | LR: {sonuclar['LR']:.6f} | Dropout: {sonuclar['Dropout']:.2f} | Batch: {sonuclar['Batch']}")
    print("#"*60)

if __name__ == "__main__":
    main()