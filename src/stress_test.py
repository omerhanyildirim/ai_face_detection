import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from model import SimpleCNN

def apply_stress(image, stress_type="none"):
    """Resme yapay bozulmalar ekler."""
    if stress_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=3)) # Bulanıklaştırma
    elif stress_type == "noise":
        # Kumlanma (Noise) ekleme
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 25, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    elif stress_type == "darken":
        # Görüntüyü karartma
        img_array = np.array(image).astype(np.float32) * 0.4
        return Image.fromarray(img_array.astype(np.uint8))
    return image

def stress_predict(image_path, model_path="best_deepfake_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_scenarios = ["original", "blur", "noise", "darken"]
    original_img = Image.open(image_path).convert("RGB")

    print(f"\n--- '{image_path}' İçin Stres Testi Başlıyor ---")
    
    for scenario in test_scenarios:
        # Bozulmayı uygula
        stressed_img = apply_stress(original_img, scenario)
        input_tensor = transform(stressed_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()

        result = "FAKE" if prob > 0.5 else "REAL"
                        # Olasılık değerini sonuca göre ayarla
        confidence = prob if prob > 0.5 else (1 - prob)
        
        print(f"[{scenario.upper()}] Durumunda Tahmin: {result} | Güven: %{confidence*100:2f}")

if __name__ == "__main__":
    # Test etmek istediğin o sahte resmin adını yaz
    stress_predict("test_yuz.jpg")