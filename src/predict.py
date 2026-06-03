import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from model import SimpleCNN, EfficientNetDeepfake

class EnsembleDeepfakeDetector:
    def __init__(self, cnn_path="best_deepfake_model.pth", effnet_path="best_effnet_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Çok Modelli (Ensemble) Sistem Başlatılıyor... Cihaz: {self.device}")
        
        try:
            self.cnn_model = SimpleCNN()
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device, weights_only=True))
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            print("✅ SimpleCNN başarıyla yüklendi.")
        except Exception as e:
            print(f"⚠️ SimpleCNN yüklenemedi: {e}")
            self.cnn_model = None

        try:
            self.effnet_model = EfficientNetDeepfake(dropout_rate=0.5) 
            self.effnet_model.load_state_dict(torch.load(effnet_path, map_location=self.device, weights_only=True))
            self.effnet_model.to(self.device)
            self.effnet_model.eval()
            print("✅ EfficientNet başarıyla yüklendi.")
        except Exception as e:
            print(f"⚠️ EfficientNet yüklenemedi: {e}")
            self.effnet_model = None

        if not self.cnn_model and not self.effnet_model:
            print("❌ Hiçbir model yüklenemedi! Lütfen .pth dosyalarınızın klasörde olduğundan emin olun.")
            exit()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.classes = {0: "FAKE (Sahte) 🚨", 1: "REAL (Gerçek) ✅"}

    def select_image_from_desktop(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("\nLütfen test etmek istediğiniz yüz görselini seçin...")
        file_path = filedialog.askopenfilename(
            title="Analiz Edilecek Yüz Görselini Seçin",
            filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")]
        )
        return file_path

    def analyze_image(self):
        image_path = self.select_image_from_desktop()
        
        if not image_path:
            print("Resim seçimi iptal edildi.")
            return

        print(f"\n🔍 Ortak Akıl Analizi Başlıyor: {os.path.basename(image_path)}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            prob_cnn = 0.0
            prob_eff = 0.0

            with torch.no_grad():
                if self.cnn_model:
                    logits_cnn = self.cnn_model(input_tensor)
                    prob_cnn = torch.sigmoid(logits_cnn).item()
                    cnn_label = "REAL" if prob_cnn > 0.5 else "FAKE"
                    print(f"  -> SimpleCNN Kararı  : {cnn_label} (Güven: %{max(prob_cnn, 1-prob_cnn)*100:.1f})")

                if self.effnet_model:
                    with torch.amp.autocast('cuda'): 
                        logits_eff = self.effnet_model(input_tensor)
                        prob_eff = torch.sigmoid(logits_eff).item()
                        eff_label = "REAL" if prob_eff > 0.5 else "FAKE"
                        print(f"  -> EfficientNet Kararı: {eff_label} (Güven: %{max(prob_eff, 1-prob_eff)*100:.1f})")

            final_prob = (prob_cnn * 0.40) + (prob_eff * 0.60)
            
            if not self.cnn_model: final_prob = prob_eff
            if not self.effnet_model: final_prob = prob_cnn

            class_idx = 1 if final_prob > 0.5 else 0
            prediction_text = self.classes[class_idx]
            
            confidence = final_prob if class_idx == 1 else (1 - final_prob)
            confidence_percent = confidence * 100

            self._show_result(image, prediction_text, confidence_percent)

        except Exception as e:
            print(f"❌ Resim analiz edilirken bir hata oluştu: {e}")

    def _show_result(self, image, label, confidence):
        print("=" * 50)
        print("🏆 NİHAİ SİSTEM KARARI (ENSEMBLE)")
        print(f"SONUÇ       : {label}")
        print(f"GÜVEN ORANI : %{confidence:.2f}")
        print("=" * 50)

        plt.figure(figsize=(7, 6))
        plt.imshow(image)
        color = 'green' if "REAL" in label else 'red'
        
        title_text = f"NİHAİ SONUÇ: {label}\nOrtak Güven: %{confidence:.2f}"
        plt.title(title_text, color=color, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.gcf().canvas.manager.set_window_title("Çok Modelli Deepfake Analizi")
        plt.show()

if __name__ == "__main__":
    detector = EnsembleDeepfakeDetector()
    detector.analyze_image()