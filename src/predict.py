import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import SimpleCNN

class DeepfakeInference:
    def __init__(self, model_path="best_deepfake_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modeli başlat ve ağırlıkları yükle
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Eğitimdeki aynı transformasyonları uygula
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Eğitim loglarına göre sınıflar: {'fake': 0, 'real': 1}
        self.classes = {0: "FAKE (Sahte) 🚨", 1: "REAL (Gerçek) ✅"}

    def run_prediction(self, image_path):
        try:
            # Görüntüyü hazırla
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                # Logit'i olasılığa çevir (0 ile 1 arası)
                probability = torch.sigmoid(logits).item()

            # Sınıf tespiti (Log: fake=0, real=1)
            # Prob > 0.5 ise Real(1), değilse Fake(0)
            class_idx = 1 if probability > 0.5 else 0
            prediction_text = self.classes[class_idx]
            
            # Güven hesaplama: 0.5'ten uzaklık (yüzde cinsinden)
            confidence = probability if class_idx == 1 else (1 - probability)
            confidence_percent = confidence * 100

            # Görselleştirme ve Sonuç
            self._display_result(image, prediction_text, confidence_percent)
            
            return class_idx, confidence_percent

        except Exception as e:
            print(f"❌ Tahmin sırasında hata oluştu: {e}")
            return None

    def _display_result(self, image, label, confidence):
        plt.figure(figsize=(6, 7))
        plt.imshow(image)
        color = 'green' if "REAL" in label else 'red'
        
        title_text = f"SONUÇ: {label}\nGüven: %{confidence:.2f}"
        plt.title(title_text, color=color, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        print(f"\n{'-'*30}")
        print(f"🖼️  Dosya: Tahmin tamamlandı.")
        print(f"📊 {title_text.replace(chr(10), ' | ')}")
        print(f"{'-'*30}")
        
        plt.show()

# --- Kullanım Örneği ---
if __name__ == "__main__":
    predictor = DeepfakeInference()
    
    # Test etmek istediğin resim yolunu buraya yaz
    
    target_image = "1.jpg" 
    
    predictor.run_prediction(target_image)