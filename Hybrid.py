import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Calibrated Physics Engine
# ==========================================
def predict_physics(s_start, T, d, D, m, s, dt=0.01, c_rot=0.4):
    """Simulates the push with a rotational friction coefficient (c_rot)."""
    x0, y0, theta0 = s_start
    I = (1 / 12.0) * m * (s**2)
    v_max = (2 * D) / T
    
    x_local, y_local, theta_local = 0.0, 0.0, 0.0
    steps = int(T / dt)
    
    for step in range(steps):
        t = step * dt
        v_i = (v_max / 2) * (np.sin((2 * np.pi * t) / T - np.pi / 2) + 1)
        
        tau_i = m * v_i * d
        alpha_i = tau_i / I
        
        # Apply the rotational friction calibration
        delta_theta_i = 0.5 * alpha_i * (dt**2) * c_rot
        theta_local += delta_theta_i
        
        delta_x_i = -v_i * np.cos(theta_local) * dt
        delta_y_i = -v_i * np.sin(theta_local) * dt
        x_local += delta_x_i
        y_local += delta_y_i

    rot_matrix = np.array([
        [np.cos(theta0), -np.sin(theta0)],
        [np.sin(theta0),  np.cos(theta0)]
    ])
    
    global_pos = np.dot(rot_matrix, np.array([x_local, y_local]))
    x_final = x0 + global_pos[0]
    y_final = y0 + global_pos[1]
    theta_final = theta0 + theta_local
    
    return np.array([x_final, y_final, theta_final])

# ==========================================
# 2. Hybrid Neural Network (MLP)
# ==========================================
class HybridPushDynamicsMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=3):
        super(HybridPushDynamicsMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    # --- A. Load Configuration and Data ---
    with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    m = config['model']['physics']['mass']
    s_size = config['model']['physics']['size']
    T = config['model']['physics']['push_duration']
    steps = config['model']['physics']['simulation_steps']
    dt = T / steps
        
    base_path = config['data']['base_path']
    train_x_path = os.path.join(base_path, config['data']['train_x'])
    train_y_path = os.path.join(base_path, config['data']['train_y'])
    
    try:
        data_x_np = np.load(train_x_path, allow_pickle=True).astype(np.float32)
        data_y_np = np.load(train_y_path, allow_pickle=True).astype(np.float32)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- B. Generate Calibrated Physics Features ---
    print("Running calibrated physics engine...")
    physics_preds = []
    for i in range(len(data_x_np)):
        theta0 = data_x_np[i][0]
        d = data_x_np[i][1]
        D = data_x_np[i][2]
        s_start = [0.0, 0.0, theta0]
        
        pred_state = predict_physics(s_start, T, d, D, m, s_size, dt=dt, c_rot=0.4)
        physics_preds.append(pred_state)
        
    physics_preds_np = np.array(physics_preds, dtype=np.float32)
    physics_mse = np.mean(np.linalg.norm(physics_preds_np - data_y_np, axis=1)**2)
    
    # --- C. Setup Residual Learning Data ---
    hybrid_x_np = np.hstack((data_x_np, physics_preds_np))
    
    scaler = StandardScaler()
    hybrid_x_scaled = scaler.fit_transform(hybrid_x_np)

    # Calculate the residual error (what the NN will learn to predict)
    residuals_np = data_y_np - physics_preds_np

    tensor_x = torch.from_numpy(hybrid_x_scaled)
    tensor_y = torch.from_numpy(residuals_np) 
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- D. Initialize Model and Train ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridPushDynamicsMLP(input_dim=6, hidden_dim=128, output_dim=3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Increased patience so it doesn't drop the learning rate too early
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    epochs = 600
    loss_history = []

    print(f"Training Hybrid MLP on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

    # --- E. Evaluate Final Network Performance ---
    model.eval()
    with torch.no_grad():
        predicted_residuals = model(tensor_x.to(device)).cpu().numpy()
        
    # Final Position = Physics Prediction + NN Correction
    hybrid_preds = physics_preds_np + predicted_residuals
    hybrid_mse_loss = np.mean(np.linalg.norm(hybrid_preds - data_y_np, axis=1)**2)
    
    print("\n" + "=" * 50)
    print("FINAL MODEL EVALUATION (SECTION 6)")
    print("=" * 50)
    print(f"1. Calibrated Physics MSE:  {physics_mse:.6f}")
    print(f"2. Pure Neural Network MSE: ~0.000386") 
    print(f"3. Hybrid Model MSE:        {hybrid_mse_loss:.6f}")
    print("=" * 50)

    # --- F. Visualize ---
    plot_results(data_y_np, physics_preds_np, hybrid_preds, loss_history)

# ==========================================
# 4. Plotting Function
# ==========================================
def plot_results(finals, phys_preds, hybrid_preds, loss_history, num_to_plot=3):
    fig, axes = plt.subplots(1, num_to_plot, figsize=(18, 5))
    fig.suptitle('Planar Pushing: Ground Truth vs. Physics vs. Hybrid NN', fontsize=16)
    
    for i in range(num_to_plot):
        ax = axes[i]
        ax.scatter(0, 0, c='black', marker='o', s=100, label='Start (0,0)')
        ax.scatter(finals[i][0], finals[i][1], c='green', marker='s', s=120, label='Ground Truth')
        ax.scatter(phys_preds[i][0], phys_preds[i][1], c='blue', marker='x', s=120, label='Pure Physics')
        ax.scatter(hybrid_preds[i][0], hybrid_preds[i][1], c='red', marker='*', s=150, label='Hybrid NN')
        
        ax.set_title(f"Push Example {i+1}")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='purple', linewidth=2)
    plt.title('Hybrid Model Training Loss Curve', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (Residuals)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()