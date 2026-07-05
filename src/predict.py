import torch
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime
from tqdm import tqdm
from model import SimpleCNN, EfficientNetDeepfake, ResNet18Deepfake, ViTDeepfake

class AcademicEnsembleBulkDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sistem Başlatılıyor. Cihaz: {self.device}")
        print("-" * 50)
        
        self.models = {}
        
        # YENİ 4'LÜ KOMİTE AĞIRLIKLARI (Ters Hata Oranına Göre)
        self.weights = {'vit': 0.38, 'eff': 0.34, 'res': 0.17, 'cnn': 0.11}

        # Modelleri Yükleme
        try:
            m = ViTDeepfake()
            m.load_state_dict(torch.load("best_vit_model.pth", map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.models['vit'] = m
            print(f"ViT yüklendi. (Ağırlık: %{self.weights['vit']*100:.0f})")
        except: print("ViT bulunamadı.")

        try:
            m = EfficientNetDeepfake()
            m.load_state_dict(torch.load("best_effnet_model.pth", map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.models['eff'] = m
            print(f"EfficientNet yüklendi. (Ağırlık: %{self.weights['eff']*100:.0f})")
        except: print("EfficientNet bulunamadı.")

        try:
            m = ResNet18Deepfake()
            m.load_state_dict(torch.load("best_resnet18_model.pth", map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.models['res'] = m
            print(f"ResNet18 yüklendi. (Ağırlık: %{self.weights['res']*100:.0f})")
        except: print("ResNet18 bulunamadı.")

        try:
            m = SimpleCNN()
            m.load_state_dict(torch.load("best_deepfake_model.pth", map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.models['cnn'] = m
            print(f"SimpleCNN yüklendi. (Ağırlık: %{self.weights['cnn']*100:.0f})")
        except: print("SimpleCNN bulunamadı.")

        print("-" * 50)
        if not self.models:
            print("Hiçbir model yüklenemedi! .pth dosyalarını kontrol ediniz.")
            exit()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.classes = {0: "FAKE (Sahte)", 1: "REAL (Gerçek)"}

    def analyze_bulk(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("\nTest etmek istediğiniz YÜZ GÖRSELLERİNİ seçin (Birden fazla seçebilirsiniz)...")
        image_paths = filedialog.askopenfilenames(
            title="Analiz İçin Görselleri Seçin", 
            filetypes=[("Resim Formatları", "*.jpg *.jpeg *.png")]
        )
        
        if not image_paths:
            print("İşlem iptal edildi. Görsel seçilmedi.")
            return

        print(f"\nToplam {len(image_paths)} adet görsel analiz ediliyor...")
        
        # Rapor için hazırlık ve sayaçlar
        zaman_etiketi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rapor_satirlari = []
        rapor_satirlari.append("="*70)
        rapor_satirlari.append(f"DEEPFAKE 4'LÜ KOMİTE - TOPLU ANALİZ RAPORU | Tarih: {zaman_etiketi}")
        rapor_satirlari.append("="*70 + "\n")

        # Toplam ve bireysel model sayaçları
        total_real = 0
        total_fake = 0
        
        # Hangi modeller yüklüyse onlar için sayaç oluştur
        model_stats = {name: {'real': 0, 'fake': 0} for name in self.models.keys()}

        try:
            for image_path in tqdm(image_paths, desc="Görseller İşleniyor"):
                dosya_adi = os.path.basename(image_path)
                image = Image.open(image_path).convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                probs = {}
                model_detaylari = []
                
                with torch.no_grad():
                    for name, model in self.models.items():
                        with torch.amp.autocast('cuda'): 
                            prob = torch.sigmoid(model(input_tensor)).item()
                            probs[name] = prob
                            
                            m_class = 'REAL' if prob > 0.5 else 'FAKE'
                            m_conf = max(prob, 1 - prob) * 100
                            model_detaylari.append(f"{name.upper():<12}: {m_class:<4} (Güven: %{m_conf:.1f})")
                            
                            # Modellerin kendi sayaçlarını güncelle
                            if prob > 0.5:
                                model_stats[name]['real'] += 1
                            else:
                                model_stats[name]['fake'] += 1

                # Ensemble Hesaplaması
                total_weight = sum([self.weights[name] for name in probs.keys()])
                final_prob = sum([(probs[name] * (self.weights[name] / total_weight)) for name in probs.keys()])

                class_idx = 1 if final_prob > 0.5 else 0
                prediction_text = self.classes[class_idx]
                confidence = final_prob if class_idx == 1 else (1 - final_prob)

                # Komite sayaçlarını güncelle
                if class_idx == 1:
                    total_real += 1
                else:
                    total_fake += 1

                # Rapora Bireysel Sonuçları Ekleme
                rapor_satirlari.append(f"► DOSYA: {dosya_adi}")
                for detay in model_detaylari:
                    rapor_satirlari.append(f"   ├─ {detay}")
                
                rapor_satirlari.append(f"   └─ KOMİTE NİHAİ KARARI : {prediction_text} (Ortak Güven: %{confidence*100:.2f})")
                rapor_satirlari.append("-" * 70)

            # --- GENEL ÖZET KISMI ---
            rapor_satirlari.append("\n" + "="*70)
            rapor_satirlari.append("GENEL ÖZET")
            rapor_satirlari.append("="*70)
            rapor_satirlari.append(f"Toplam İncelenen Görsel : {len(image_paths)}")
            rapor_satirlari.append("-" * 70)
            
            # Alt Modellerin Özeti
            rapor_satirlari.append("--- BİREYSEL MODEL KARARLARI ---")
            for name in self.models.keys():
                rapor_satirlari.append(f"[{name.upper():<12}] -> GERÇEK (REAL): {model_stats[name]['real']:<4} | SAHTE (FAKE): {model_stats[name]['fake']:<4}")
            
            rapor_satirlari.append("-" * 70)
            
            # Nihai Komite Özeti
            rapor_satirlari.append("--- KOMİTE (ENSEMBLE) KARARI ---")
            rapor_satirlari.append(f"Toplam GERÇEK (REAL) : {total_real}")
            rapor_satirlari.append(f"Toplam SAHTE (FAKE)  : {total_fake}")
            rapor_satirlari.append("="*70 + "\n")

            # Raporu TXT dosyasına yazma
            rapor_dosyasi = "toplu_analiz_raporu.txt"
            with open(rapor_dosyasi, "w", encoding="utf-8") as f:
                f.write("\n".join(rapor_satirlari))
            
            print("\n" + "=" * 50)
            print("ANALİZ TAMAMLANDI!")
            print(f"Sonuçlar '{rapor_dosyasi}' dosyasına kaydedildi.")
            print("=" * 50)

        except Exception as e: 
            print(f"\nAnaliz sırasında bir hata oluştu..! {e}")

if __name__ == "__main__":
    detector = AcademicEnsembleBulkDetector()
    detector.analyze_bulk()