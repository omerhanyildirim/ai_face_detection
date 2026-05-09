import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from dataset import get_data_loaders
from model import SimpleCNN

# ==========================================
# PSO (Parçacık Sürüsü Optimizasyonu) Sınıfı
# ==========================================
class Particle:
    def __init__(self, bounds):
        # Konumlar: [0]: LR (Log scale), [1]: Dropout, [2]: Batch Index
        self.position = [
            random.uniform(bounds[0][0], bounds[0][1]), # LR
            random.uniform(bounds[1][0], bounds[1][1]), # Dropout
            random.uniform(bounds[2][0], bounds[2][1])  # Batch Size Index
        ]
        # Hızlar
        self.velocity = [random.uniform(-1, 1) for _ in range(len(bounds))]
        
        # En iyi kişisel hafıza
        self.best_position = list(self.position)
        self.best_loss = float('inf')
        self.current_loss = float('inf')

def evaluate_particle(position, device):
    """Parçacığın temsil ettiği hiperparametreleri modele verip 2 epoch eğitir ve test eder."""
    # 1. Pozisyonları gerçek hiperparametre değerlerine dönüştür (Decode)
    lr = 10 ** position[0]  # Örn: -3 -> 0.001
    dropout = position[1]
    
    # Batch size'ı listeye eşle [32, 64, 128]
    batch_options = [32, 64, 128]
    batch_idx = int(round(position[2]))
    # Sınırları aşmasını engelle
    batch_idx = max(0, min(2, batch_idx)) 
    batch = batch_options[batch_idx]

    print(f"  -> Deneniyor: LR={lr:.5f}, Dropout={dropout:.2f}, Batch={batch}")

    # 2. Veri Yükleyici ve Modeli Hazırla
    train_loader, valid_loader = get_data_loaders(batch_size=batch)
    if train_loader is None: return float('inf')

    model = SimpleCNN(dropout_rate=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Hızlı değerlendirme için sadece 2 Epoch
    for epoch in range(2):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                predictions = model(images)
                valid_loss_sum += criterion(predictions, labels).item()

    return valid_loss_sum / len(valid_loader)

def run_pso():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- PSO Hiperparametre Optimizasyonu Başlıyor ({device}) ---\n")

    # PSO Sınırları: [(min_lr_log, max_lr_log), (min_drop, max_drop), (min_batch_idx, max_batch_idx)]
    bounds = [(-4.0, -2.0), (0.2, 0.6), (0.0, 2.0)]
    
    # PSO Hiperparametreleri
    num_particles = 5    # Sürüdeki kuş sayısı
    num_iterations = 4   # Tur sayısı
    w = 0.5              # Eylemsizlik (Inertia - Mevcut hızı koruma eğilimi)
    c1 = 1.5             # Bilişsel Katsayı (Kendi en iyisine yönelme)
    c2 = 1.5             # Sosyal Katsayı (Sürünün en iyisine yönelme)

    # Sürüyü oluştur
    swarm = [Particle(bounds) for _ in range(num_particles)]
    
    global_best_position = []
    global_best_loss = float('inf')

    for iteration in range(num_iterations):
        print(f"\n================ ITERASYON {iteration + 1}/{num_iterations} ================")
        
        # 1. Her parçacığın performansını ölç (Fitness Evaluation)
        for i, particle in enumerate(swarm):
            print(f"Parçacık {i+1}:")
            loss = evaluate_particle(particle.position, device)
            particle.current_loss = loss
            print(f"  -> Sonuç Kayıp (Loss): {loss:.4f}\n")

            # Kişisel en iyiyi (pbest) güncelle
            if loss < particle.best_loss:
                particle.best_loss = loss
                particle.best_position = list(particle.position)

            # Sürünün en iyisini (gbest) güncelle
            if loss < global_best_loss:
                global_best_loss = loss
                global_best_position = list(particle.position)

        # 2. Hızları ve Pozisyonları Güncelle (PSO Matematiği)
        for particle in swarm:
            for d in range(len(bounds)):
                r1, r2 = random.random(), random.random()
                
                # Yeni Hız: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                cognitive = c1 * r1 * (particle.best_position[d] - particle.position[d])
                social = c2 * r2 * (global_best_position[d] - particle.position[d])
                particle.velocity[d] = (w * particle.velocity[d]) + cognitive + social
                
                # Yeni Pozisyon: x = x + v
                particle.position[d] += particle.velocity[d]
                
                # Sınır kontrolü (Uzay dışına çıkarlarsa sınıra çek)
                if particle.position[d] < bounds[d][0]:
                    particle.position[d] = bounds[d][0]
                elif particle.position[d] > bounds[d][1]:
                    particle.position[d] = bounds[d][1]

    # Sonuçları Çözümle
    best_lr = 10 ** global_best_position[0]
    best_drop = global_best_position[1]
    best_batch = [32, 64, 128][int(round(max(0, min(2, global_best_position[2]))))]

    print("\n" + "*"*50)
    print("🏆 PSO OPTİMİZASYONU TAMAMLANDI!")
    print(f"En Düşük Validasyon Kaybı: {global_best_loss:.4f}")
    print("EN İYİ HİPERPARAMETRELER (Bunları train.py'ye girin):")
    print(f"  - Learning Rate : {best_lr:.6f}")
    print(f"  - Dropout Rate  : {best_drop:.2f}")
    print(f"  - Batch Size    : {best_batch}")
    print("*"*50)

if __name__ == "__main__":
    run_pso()