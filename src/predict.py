import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from model import SimpleCNN, EfficientNetDeepfake, ResNet18Deepfake

class AcademicEnsembleDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sistem Başlatılıyor. Cihaz: {self.device}")
        print("-" * 50)
        
        self.models = {}
        self.weights = {'res': 0.55, 'eff': 0.30, 'cnn': 0.15}

        # 1. ResNet18 (%55)
        try:
            m = ResNet18Deepfake()
            m.load_state_dict(torch.load("best_resnet18_model.pth", map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.models['res'] = m
            print(f"ResNet18 yüklendi. (Ağırlık: %{self.weights['res']*100})")
        except: print("ResNet18 bulunamadı.")

        # 2. EfficientNet (%30)
        try:
            m = EfficientNetDeepfake()
            m.load_state_dict(torch.load("best_effnet_model.pth", map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.models['eff'] = m
            print(f"EfficientNet yüklendi. (Ağırlık: %{self.weights['eff']*100})")
        except: print("EfficientNet bulunamadı.")

        # 3. SimpleCNN (%15)
        try:
            m = SimpleCNN()
            m.load_state_dict(torch.load("best_deepfake_model.pth", map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.models['cnn'] = m
            print(f"SimpleCNN yüklendi. (Ağırlık: %{self.weights['cnn']*100})")
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

    def analyze(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("\nTest etmek istediğiniz yüz görselini seçin...")
        image_path = filedialog.askopenfilename(title="Analiz İçin Görsel Seçin", filetypes=[("Resim", "*.jpg *.jpeg *.png")])
        
        if not image_path:
            print("İşlem iptal edildi.")
            return

        print(f"\nAnaliz yapılıyor: {os.path.basename(image_path)}")
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            probs = {}
            with torch.no_grad():
                for name, model in self.models.items():
                    with torch.amp.autocast('cuda'): 
                        prob = torch.sigmoid(model(input_tensor)).item()
                        probs[name] = prob
                        print(f"  -> {name.upper():<4} Kararı: {'REAL' if prob>0.5 else 'FAKE'} (%{max(prob, 1-prob)*100:.1f})")

            total_weight = sum([self.weights[name] for name in probs.keys()])
            final_prob = sum([(probs[name] * (self.weights[name] / total_weight)) for name in probs.keys()])

            class_idx = 1 if final_prob > 0.5 else 0
            prediction_text = self.classes[class_idx]
            confidence = final_prob if class_idx == 1 else (1 - final_prob)

            print("=" * 50)
            print(f"KARAR: {prediction_text} (Güven: %{confidence*100:.2f})")
            print("=" * 50)

            plt.figure(figsize=(7, 6))
            plt.imshow(image)
            plt.title(f"NİHAİ SONUÇ: {prediction_text}\nOrtak Güven: %{confidence*100:.2f}", color='green' if class_idx==1 else 'red', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.gcf().canvas.manager.set_window_title("Çoklu model karar komisyonu")
            plt.show()

        except Exception as e: 
            print(f"Analiz sırasında bir hata oluştu..! {e}")

if __name__ == "__main__":
    detector = AcademicEnsembleDetector()
    detector.analyze()