import torch
import torch.nn as nn
import torch.optim as optim
import random
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

    num_epochs = 2 
    PSO_TRAIN_SAMPLES = 3000  
    PSO_VALID_SAMPLES = 1000  
    
    max_train_steps = max(1, PSO_TRAIN_SAMPLES // batch)
    max_valid_steps = max(1, PSO_VALID_SAMPLES // batch)

    for epoch in range(num_epochs):
        model.train()
        train_loop = tqdm(train_loader, desc=f"  [PSO - {model_name} Ep {epoch+1}] Eğitim (Mini-Set)", leave=False, total=max_train_steps)
        
        for step, (images, labels) in enumerate(train_loop):
            if step >= max_train_steps: break 
                
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, labels)
                
            loss.backward()
            optimizer.step()
            train_loop.set_postfix(loss=loss.item())

        model.eval()
        valid_loss_sum = 0
        valid_loop = tqdm(valid_loader, desc=f"  [PSO - {model_name} Ep {epoch+1}] Test (Mini-Set)", leave=False, total=max_valid_steps)
        
        with torch.no_grad():
            for step, (images, labels) in enumerate(valid_loop):
                if step >= max_valid_steps: break
                    
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(images)
                    loss = criterion(predictions, labels)
                    
                valid_loss_sum += loss.item()

    return valid_loss_sum / max_valid_steps

def run_pso(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- PSO başlatılıyor: {model_name} ({device}) ---")

    if model_name in ["EfficientNet", "ResNet18"]:
        bounds = [(-5.0, -3.0), (0.2, 0.6), (0.0, 2.0)] 
    else: 
        bounds = [(-4.0, -2.0), (0.2, 0.6), (0.0, 2.0)] 
    
    num_particles = 5    
    num_iterations = 4   
    w, c1, c2 = 0.5, 1.5, 1.5             

    swarm = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = []
    global_best_loss = float('inf')

    for iteration in range(num_iterations):
        for i, particle in enumerate(swarm):
            loss = evaluate_particle(particle.position, device, model_name)
            particle.current_loss = loss

            if loss < particle.best_loss:
                particle.best_loss = loss
                particle.best_position = list(particle.position)
            if loss < global_best_loss:
                global_best_loss = loss
                global_best_position = list(particle.position)

        for particle in swarm:
            for d in range(len(bounds)):
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (particle.best_position[d] - particle.position[d])
                social = c2 * r2 * (global_best_position[d] - particle.position[d])
                particle.velocity[d] = (w * particle.velocity[d]) + cognitive + social
                particle.position[d] += particle.velocity[d]
                particle.position[d] = max(bounds[d][0], min(particle.position[d], bounds[d][1]))

    best_lr = 10 ** global_best_position[0]
    best_drop = global_best_position[1]
    best_batch = [32, 64, 128][int(round(max(0, min(2, global_best_position[2]))))]

    print(f"{model_name} PSO tamamlandı! -> LR: {best_lr:.5f}, Drop: {best_drop:.2f}, Batch: {best_batch}")
    return best_lr, best_drop, best_batch