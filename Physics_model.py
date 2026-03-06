import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Physics Prediction Function
# ==========================================
def predict_physics(s_start, T, d, D, m, s, dt=0.01, c_rot=0.4): 
    """Simulates the push and predicts the final state of the object."""
    x0, y0, theta0 = s_start
    I = (1 / 12.0) * m * (s**2)
    v_max = (2 * D) / T
    
    x_local, y_local, theta_local = 0.0, 0.0, 0.0
    steps = int(T / dt)
    
    trajectory_local = [[0.0, 0.0]]
    
    for step in range(steps):
        t = step * dt
        v_i = (v_max / 2) * (np.sin((2 * np.pi * t) / T - np.pi / 2) + 1)
        
        # Angular Update
        tau_i = m * v_i * d
        alpha_i = tau_i / I
        
        # MULTIPLY BY c_rot TO SIMULATE TABLE ROTATIONAL FRICTION
        delta_theta_i = 0.5 * alpha_i * (dt**2) * c_rot
        theta_local += delta_theta_i
        
        # Position Update
        delta_x_i = -v_i * np.cos(theta_local) * dt
        delta_y_i = -v_i * np.sin(theta_local) * dt
        x_local += delta_x_i
        y_local += delta_y_i
        
        trajectory_local.append([x_local, y_local])

    rot_matrix = np.array([
        [np.cos(theta0), -np.sin(theta0)],
        [np.sin(theta0),  np.cos(theta0)]
    ])
    
    # Transform to global coordinates
    local_pos = np.array([x_local, y_local])
    global_pos = np.dot(rot_matrix, local_pos)
    
    x_final = x0 + global_pos[0]
    y_final = y0 + global_pos[1]
    theta_final = theta0 + theta_local
    
    # Convert entire trajectory for plotting
    trajectory_global = np.dot(rot_matrix, np.array(trajectory_local).T).T
    trajectory_global[:, 0] += x0
    trajectory_global[:, 1] += y0
    
    return np.array([x_final, y_final, theta_final]), trajectory_global

# ==========================================
# 2. Main Execution and Evaluation
# ==========================================
def main():
    # --- A. Load Configuration ---
    with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    m = config['model']['physics']['mass']
    s = config['model']['physics']['size']
    T = config['model']['physics']['push_duration']
    steps = config['model']['physics']['simulation_steps']
    dt = T / steps

    # --- B. Load Dataset ---
    print("Loading data...")
    base_path = config['data']['base_path']
    train_x_path = os.path.join(base_path, config['data']['train_x'])
    train_y_path = os.path.join(base_path, config['data']['train_y'])
    
    try:
        data_x = np.load(train_x_path, allow_pickle=True).astype(np.float32)
        data_y = np.load(train_y_path, allow_pickle=True).astype(np.float32)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- C. Run the Simulation ---
    predictions = []
    trajectories = [] 
    starts = [] 
    
    print(f"Running calibrated physics simulation for {len(data_x)} pushes...")
    
    for i in range(len(data_x)):
        theta0 = data_x[i][0]
        d = data_x[i][1]
        D = data_x[i][2]
        
        s_start = [0.0, 0.0, theta0]
        starts.append(s_start) 
        
        # Call the physics function (c_rot=0.4 is set as the default in the function definition)
        pred_state, traj = predict_physics(s_start, T, d, D, m, s, dt=dt)
        predictions.append(pred_state)
        trajectories.append(traj) 
        
    predictions = np.array(predictions)
    
    # --- D. Calculate MSE Loss ---
    squared_errors = np.linalg.norm(predictions - data_y, axis=1)**2
    mse_loss = np.mean(squared_errors)
    
    print("-" * 40)
    print(f"Calibrated Physics Model MSE Loss: {mse_loss:.6f}")
    print("-" * 40)

    # --- E. Plot Results ---
    num_to_plot = min(3, len(data_x))
    plot_results(starts[:num_to_plot], data_y[:num_to_plot], predictions[:num_to_plot], trajectories[:num_to_plot])

# ==========================================
# 3. Plotting Function
# ==========================================
def plot_results(starts, finals, preds, trajs):
    fig, axes = plt.subplots(1, len(starts), figsize=(15, 5))
    fig.suptitle('Calibrated Physics Model Predictions vs Ground Truth', fontsize=14)
    
    for i in range(len(starts)):
        ax = axes[i]
        
        ax.scatter(starts[i][0], starts[i][1], c='blue', marker='o', s=100, label='Start')
        ax.scatter(finals[i][0], finals[i][1], c='green', marker='s', s=100, label='Ground Truth Final')
        ax.scatter(preds[i][0], preds[i][1], c='red', marker='X', s=100, label='Predicted Final')
        
        traj = trajs[i]
        ax.plot(traj[:, 0], traj[:, 1], 'r--', label='Predicted Path')
        
        ax.set_title(f"Push Example {i+1}")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.grid(True)
        ax.axis('equal') 
        
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()