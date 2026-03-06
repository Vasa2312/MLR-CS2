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
# 1. Define the Neural Network (MLP)
# ==========================================
class PushDynamicsMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=180, output_dim=3):
        super(PushDynamicsMLP, self).__init__()
        # At least one hidden layer (we use two for better capacity)
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
# 2. Main Execution
# ==========================================
def main():
    
    # --- A. Load Configuration and Data ---
    with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    base_path = config['data']['base_path']
    train_x_path = os.path.join(base_path, config['data']['train_x'])
    train_y_path = os.path.join(base_path, config['data']['train_y'])
    
    try:
        data_x_np = np.load(train_x_path, allow_pickle=True).astype(np.float32)
        data_y_np = np.load(train_y_path, allow_pickle=True).astype(np.float32)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- B. Normalize Inputs (X only) ---
    scaler = StandardScaler()
    data_x_scaled = scaler.fit_transform(data_x_np)

    # --- C. Prepare PyTorch Tensors & DataLoader ---
    # Now we create the tensors using the scaled X, and the unscaled Y
    tensor_x = torch.from_numpy(data_x_scaled).float()
    tensor_y = torch.from_numpy(data_y_np).float()
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- D. Initialize Model, Loss, and Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PushDynamicsMLP(input_dim=3, hidden_dim=180, output_dim=3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # The scheduler will drop the learning rate if the loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    epochs = 600
    loss_history = []

    print(f"Training MLP on {device} for {epochs} epochs...")

    # --- D. Training Loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        # Calculate average loss for the epoch
        epoch_loss /= len(dataloader.dataset)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

    # --- E. Evaluate Final Network Performance ---
    model.eval()
    with torch.no_grad():
        all_preds = model(tensor_x.to(device)).cpu().numpy()
        
    # Compute final MSE across the entire dataset
    mlp_mse_loss = np.mean(np.linalg.norm(all_preds - data_y_np, axis=1)**2)
    
    print("-" * 40)
    print(f"Neural Network Final MSE Loss: {mlp_mse_loss:.6f}")
    print("-" * 40)

    # --- F. Visualize Loss Curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss', color='blue', linewidth=2)
    plt.title('MLP Training Loss Curve (Push Dynamics)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()