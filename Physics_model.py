import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Physics Prediction Function
# ==========================================
def predict_physics(s_start, T, d, D, m, s, dt=0.01):
    """
    Simulates the push and predicts the final state of the object.
    """
    x0, y0, theta0 = s_start
    
    # Constants
    I = (1 / 12.0) * m * (s**2)
    v_max = (2 * D) / T
    
    # Local variables for integration
    x_local = 0.0
    y_local = 0.0
    theta_local = 0.0
    
    steps = int(T / dt)
    
    # Track the trajectory for visualization
    trajectory_local = [[0.0, 0.0]]
    
    # Numerical Integration
    for step in range(steps):
        t = step * dt
        
        # Velocity Profile
        v_i = (v_max / 2) * (np.sin((2 * np.pi * t) / T - np.pi / 2) + 1)
        
        # Angular Update
        tau_i = m * v_i * d
        alpha_i = tau_i / I
        
        delta_theta_i = 0.5 * alpha_i * (dt**2)
        theta_local += delta_theta_i
        
        # Position Update (Local Frame)
        delta_x_i = -v_i * np.cos(theta_local) * dt
        delta_y_i = -v_i * np.sin(theta_local) * dt
        
        x_local += delta_x_i
        y_local += delta_y_i
        
        trajectory_local.append([x_local, y_local])

    # Frame Transformation (Local to Global)
    rot_matrix = np.array([
        [np.cos(theta0), -np.sin(theta0)],
        [np.sin(theta0),  np.cos(theta0)]
    ])
    
    # Convert local final position to global position
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
    # import os is already at the top of the file, no need to repeat it here.
    
    # --- A. Load Configuration ---
    with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    m = config['model']['physics']['mass']
    s = config['model']['physics']['size']
    
    # --- B. Load Dataset ---
    print("Loading data...")
    base_path = config['data']['base_path']
    train_x_path = os.path.join(base_path, config['data']['train_x'])
    train_y_path = os.path.join(base_path, config['data']['train_y'])
    
    try:
        data_x = np.load(train_x_path, allow_pickle=True)
        data_y = np.load(train_y_path, allow_pickle=True)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- C. Setup Physics Loop Variables ---
    # FIXED: Indentation is now properly aligned with the main function scope
    T = config['model']['physics']['push_duration']
    steps = config['model']['physics']['simulation_steps']
    dt = T / steps
    
    predictions = []
    trajectories = []  # Added to store trajectory data for plotting
    starts = []        # Added to store start positions for plotting
    
    print(f"Running physics simulation for {len(data_x)} pushes...")
    
    # --- D. Run the Simulation ---
    for i in range(len(data_x)):
        theta0 = data_x[i][0]
        d = data_x[i][1]
        D = data_x[i][2]
        
        s_start = [0.0, 0.0, theta0]
        starts.append(s_start) # Save the start state
        
        # FIXED: Capture the trajectory variable instead of using '_'
        pred_state, traj = predict_physics(s_start, T, d, D, m, s, dt=dt)
        predictions.append(pred_state)
        trajectories.append(traj) # Save the trajectory
        
    predictions = np.array(predictions)
    
    # --- E. Calculate MSE Loss ---
    squared_errors = np.linalg.norm(predictions - data_y, axis=1)**2
    mse_loss = np.mean(squared_errors)
    
    print("-" * 40)
    print(f"Physics Model MSE Loss: {mse_loss:.6f}")
    print("-" * 40)

    # FIXED: Call the plotting function to visualize the results
    # We pass in a slice (e.g., first 3 examples) so the plot isn't too crowded
    num_to_plot = min(3, len(data_x))
    plot_results(starts[:num_to_plot], data_y[:num_to_plot], predictions[:num_to_plot], trajectories[:num_to_plot])

# ==========================================
# 3. Plotting Function (For your report)
# ==========================================
def plot_results(starts, finals, preds, trajs):
    """
    Plots the ground truth vs predicted positions to analyze model performance.
    """
    fig, axes = plt.subplots(1, len(starts), figsize=(15, 5))
    
    for i in range(len(starts)):
        ax = axes[i]
        
        # Plot start position
        ax.scatter(starts[i][0], starts[i][1], c='blue', marker='o', s=100, label='Start')
        
        # Plot ground truth final position
        ax.scatter(finals[i][0], finals[i][1], c='green', marker='s', s=100, label='Ground Truth Final')
        
        # Plot predicted final position
        ax.scatter(preds[i][0], preds[i][1], c='red', marker='X', s=100, label='Predicted Final')
        
        # Plot the predicted continuous trajectory
        traj = trajs[i]
        ax.plot(traj[:, 0], traj[:, 1], 'r--', label='Predicted Path')
        
        ax.set_title(f"Push Example {i+1}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)
        ax.legend()
        ax.axis('equal') # Keeps the aspect ratio accurate to real world

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()