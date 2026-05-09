import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from model import SimpleCNN
import os

class DesktopDeepfakeDetector:
    def __init__(self, model_path="best_deepfake_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Sistem Başlatılıyor... Cihaz: {self.device}")
        
        # Modeli yükle
        try:
            self.model = SimpleCNN()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model ağırlıkları başarıyla yüklendi.")
        except Exception as e:
            print(f"❌ Model yüklenirken hata oluştu: {e}")
            print("Lütfen 'best_deepfake_model.pth' dosyasının aynı klasörde olduğundan emin olun.")
            exit()

        # Test için "temiz" ön işlemler (Data augmentation BURADA OLMAZ)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.classes = {0: "FAKE (Sahte) 🚨", 1: "REAL (Gerçek) ✅"}

    def select_image_from_desktop(self):
        """Kullanıcıya dosya seçme penceresi açar"""
        root = tk.Tk()
        root.withdraw() # Ana Tkinter penceresini gizle, sadece diyalog görünsün
        root.attributes('-topmost', True) # Pencereyi en öne getir
        
        print("\n📂 Lütfen test etmek istediğiniz resmi seçin...")
        file_path = filedialog.askopenfilename(
            title="Analiz Edilecek Yüz Görselini Seçin",
            filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")]
        )
        
        return file_path

    def analyze_image(self):
        image_path = self.select_image_from_desktop()
        
        if not image_path:
            print("⚠️ Resim seçimi iptal edildi.")
            return

        print(f"\n🔍 Analiz Ediliyor: {os.path.basename(image_path)}")
        
        try:
            # Resmi aç ve modele uygun hale getir
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Tahmin yap
            with torch.no_grad():
                logits = self.model(input_tensor)
                probability = torch.sigmoid(logits).item()

            # prob > 0.5 ise Real, değilse Fake
            class_idx = 1 if probability > 0.5 else 0
            prediction_text = self.classes[class_idx]
            
            # Güven oranını hesapla
            confidence = probability if class_idx == 1 else (1 - probability)
            confidence_percent = confidence * 100

            # Sonucu hem konsola yaz hem ekranda göster
            self._show_result(image, prediction_text, confidence_percent)

        except Exception as e:
            print(f"❌ Resim analiz edilirken bir hata oluştu: {e}")

    def _show_result(self, image, label, confidence):
        # Konsol Çıktısı
        print("=" * 40)
        print(f"🎯 TESPİT SONUCU : {label}")
        print(f"📊 GÜVEN ORANI   : %{confidence:.2f}")
        print("=" * 40)

        # Görsel Çıktı (Ekranda Resim Açılsın)
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        color = 'green' if "REAL" in label else 'red'
        
        title_text = f"SONUÇ: {label}\nGüven: %{confidence:.2f}"
        plt.title(title_text, color=color, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Pencere başlığı
        plt.gcf().canvas.manager.set_window_title("Deepfake Analiz Sonucu")
        plt.show()

if __name__ == "__main__":
    detector = DesktopDeepfakeDetector()
    detector.analyze_image()