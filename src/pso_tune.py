import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gc
from dataset import get_data_loaders
from model import SimpleCNN, EfficientNetDeepfake, ResNet18Deepfake
from tqdm import tqdm

class Particle:
    def __init__(self, bounds):
        self.position = [
            random.uniform(bounds[0][0], bounds[0][1]), 
            random.uniform(bounds[1][0], bounds[1][1]), 
            random.uniform(bounds[2][0], bounds[2][1])  
        ]
        self.velocity = [random.uniform(-1, 1) for _ in range(len(bounds))]
        self.best_position = list(self.position)
        self.best_loss = float('inf')
        self.current_loss = float('inf')

def evaluate_particle(position, device, model_name):
    lr = 10 ** position[0]  
    dropout = position[1]
    
    batch_options = [32, 64, 128]
    batch_idx = int(round(position[2]))
    batch_idx = max(0, min(2, batch_idx)) 
    batch = batch_options[batch_idx]

    train_loader, valid_loader = get_data_loaders(batch_size=batch)
    if train_loader is None: return float('inf')

    if model_name == "SimpleCNN":
        model = SimpleCNN(dropout_rate=dropout).to(device)
    elif model_name == "EfficientNet":
        model = EfficientNetDeepfake(dropout_rate=dropout).to(device)
    elif model_name == "ResNet18":
        model = ResNet18Deepfake(dropout_rate=dropout).to(device)
    else:
        raise ValueError(f"Bilinmeyen model türü: {model_name}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ALTIN ORAN PARAMETRELERİ
    num_epochs = 2 
    PSO_TRAIN_SAMPLES = 4000  
    PSO_VALID_SAMPLES = 1500  
    
    max_train_steps = max(1, PSO_TRAIN_SAMPLES // batch)
    max_valid_steps = max(1, PSO_VALID_SAMPLES // batch)

    for epoch in range(num_epochs):
        model.train()
        train_loop = tqdm(train_loader, desc=f"    [Ep {epoch+1}] Eğitim", leave=False, position=1, total=max_train_steps)
        
        for step, (images, labels) in enumerate(train_loop):
            if step >= max_train_steps: break 
                
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, labels)
                
            loss.backward()
            optimizer.step()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        valid_loss_sum = 0
        valid_loop = tqdm(valid_loader, desc=f"    [Ep {epoch+1}] Test ", leave=False, position=1, total=max_valid_steps)
        
        with torch.no_grad():
            for step, (images, labels) in enumerate(valid_loop):
                if step >= max_valid_steps: break
                    
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(images)
                    loss = criterion(predictions, labels)
                    
                valid_loss_sum += loss.item()

    final_loss = valid_loss_sum / max_valid_steps

    # --- KRİTİK GPU TEMİZLİĞİ ---
    del model
    del optimizer
    del train_loader
    del valid_loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return final_loss

def run_pso(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- PSO Başlatılıyor: {model_name} ({device}) ---\n")

    if model_name in ["EfficientNet", "ResNet18"]:
        bounds = [(-5.0, -3.0), (0.2, 0.6), (0.0, 2.0)] 
    else: 
        bounds = [(-4.0, -2.0), (0.2, 0.6), (0.0, 2.0)] 
    
    num_particles = 8    
    num_iterations = 6   
    w_max, w_min = 0.9, 0.4        
    c1, c2 = 1.496, 1.496          

    checkpoint_file = f"pso_kayit_{model_name}.pkl"
    start_iteration = 0
    start_particle = 0

    # --- KAYIT (CHECKPOINT) KONTROLÜ ---
    if os.path.exists(checkpoint_file):
        print(f"[*] {model_name} için önceki bir kayıt bulundu! Yükleniyor...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            swarm = checkpoint_data['swarm']
            global_best_position = checkpoint_data['global_best_position']
            global_best_loss = checkpoint_data['global_best_loss']
            start_iteration = checkpoint_data['iteration']
            start_particle = checkpoint_data['particle'] + 1 # Bir sonraki parçacıktan başla

        # Eğer o iterasyonun son parçacığı bitmişse, bir sonraki iterasyona geç
        if start_particle >= num_particles:
            start_iteration += 1
            start_particle = 0

        # Eğer tüm iterasyonlar bitmişse işlemi kapat
        if start_iteration >= num_iterations:
            print(f"[*] {model_name} optimizasyonu zaten tamamlanmış. Sonuçlar getiriliyor...\n")
            best_lr = 10 ** global_best_position[0]
            best_drop = global_best_position[1]
            best_batch = [32, 64, 128][int(round(max(0, min(2, global_best_position[2]))))]
            return best_lr, best_drop, best_batch
        
        print(f"[*] İterasyon {start_iteration+1}/{num_iterations} | Parçacık {start_particle+1} seviyesinden devam ediliyor.\n")
    else:
        swarm = [Particle(bounds) for _ in range(num_particles)]
        global_best_position = []
        global_best_loss = float('inf')

    total_evaluations = num_iterations * num_particles
    initial_pbar_value = (start_iteration * num_particles) + start_particle
    
    pso_pbar = tqdm(total=total_evaluations, initial=initial_pbar_value, desc=f"Genel İlerleme ({model_name})", position=0, leave=True, 
                    bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} İşlem [Geçen: {elapsed} < Kalan: {remaining}]")

    for iteration in range(start_iteration, num_iterations):
        w = w_max - ((w_max - w_min) * (iteration / num_iterations))

        # Eğer kaldığı yerden başlıyorsa, for döngüsünü o parçacıktan başlatırız
        p_start_idx = start_particle if iteration == start_iteration else 0

        for i in range(p_start_idx, num_particles):
            pso_pbar.set_description(f"İterasyon {iteration+1}/{num_iterations} (Parçacık {i+1}/{num_particles})")
            
            # 1. Değerlendir
            loss = evaluate_particle(swarm[i].position, device, model_name)
            swarm[i].current_loss = loss

            # 2. En iyi durumları güncelle
            if loss < swarm[i].best_loss:
                swarm[i].best_loss = loss
                swarm[i].best_position = list(swarm[i].position)
            if loss < global_best_loss:
                global_best_loss = loss
                global_best_position = list(swarm[i].position)
            
            # 3. Hız ve Pozisyonu Hemen Güncelle (Asenkron PSO)
            for d in range(len(bounds)):
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (swarm[i].best_position[d] - swarm[i].position[d])
                social = c2 * r2 * (global_best_position[d] - swarm[i].position[d])
                swarm[i].velocity[d] = (w * swarm[i].velocity[d]) + cognitive + social
                swarm[i].position[d] += swarm[i].velocity[d]
                swarm[i].position[d] = max(bounds[d][0], min(swarm[i].position[d], bounds[d][1]))

            pso_pbar.update(1)

            # 4. HER BİR PARÇACIKTAN SONRA ANINDA KAYDET (Micro-Checkpointing)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'iteration': iteration,
                    'particle': i,
                    'swarm': swarm,
                    'global_best_position': global_best_position,
                    'global_best_loss': global_best_loss
                }, f)

    pso_pbar.close()

    best_lr = 10 ** global_best_position[0]
    best_drop = global_best_position[1]
    best_batch = [32, 64, 128][int(round(max(0, min(2, global_best_position[2]))))]

    print("\n" + "="*50)
    print(f"{model_name.upper()} İÇİN PSO TAMAMLANDI!")
    print(f"En İyi Learning Rate : {best_lr:.6f}")
    print(f"En İyi Dropout       : {best_drop:.2f}")
    print(f"En İyi Batch Size    : {best_batch}")
    print("="*50)
    
    return best_lr, best_drop, best_batch

if __name__ == "__main__":
    modeller = ["SimpleCNN", "EfficientNet", "ResNet18"]
    pso_sonuclari = {}

    print("*"*60)
    print("TÜM MODELLER İÇİN DENGELİ PSO OPTİMİZASYONU BAŞLIYOR")
    print("Not: İşlemi durdurmak için CTRL+C yapabilirsiniz. Her parçacıktan sonra kayıt alınır.")
    print("*"*60)

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
    print("Not: Bu parametreleri train.py dosyanızda kullanarak asıl eğitime başlayabilirsiniz.")