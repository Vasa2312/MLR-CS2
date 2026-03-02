import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Physics Prediction Function
# ==========================================
def predict_physics(s_start, T, d, D, m, s, dt=0.01,
                    angular_scale=77.12, asym_sin1=-2.0, asym_cos1=0.56,
                    asym_sin2=-1.07, asym_cos2=0.24):
    """
    Predicts the object motion from a push using clarification equations
    with asymmetry correction for non-square object geometry.

    Args:
        s_start: [x0, y0, theta0] initial state
        T: push duration (seconds)
        d: contact point offset (side)
        D: total push distance
        m: object mass
        s: object size
        dt: timestep for numerical integration
        angular_scale: scaling factor for offset-driven rotation
        asym_sin1: sin(theta0) correction coefficient
        asym_cos1: cos(theta0) correction coefficient
        asym_sin2: sin(2*theta0) correction coefficient
        asym_cos2: cos(2*theta0) correction coefficient

    Returns:
        predicted state [dx_global, dy_global, delta_theta],
        trajectory in global frame
    """
    x0, y0, theta0 = s_start

    v_max = (2 * D) / T

    # Local frame variables
    x_local = 0.0
    y_local = 0.0
    theta_local = 0.0

    # Asymmetry correction for non-square object geometry
    asym_rate = (asym_sin1 * np.sin(theta0) + asym_cos1 * np.cos(theta0)
                 + asym_sin2 * np.sin(2 * theta0) + asym_cos2 * np.cos(2 * theta0))

    steps = int(T / dt)
    trajectory_local = [[0.0, 0.0]]

    # Numerical integration (clarification equations + asymmetry)
    for step in range(steps):
        t = step * dt

        # 1. Velocity profile
        v_i = (v_max / 2) * (np.sin((2 * np.pi * t) / T - np.pi / 2) + 1)

        # 2. Angular update: dtheta = (K*o + asymmetry) * v * dt
        theta_local += (angular_scale * d + asym_rate) * v_i * dt

        # 3. Position update (local frame)
        x_local += -v_i * np.cos(theta_local) * dt
        y_local += -v_i * np.sin(theta_local) * dt

        trajectory_local.append([x_local, y_local])

    # 4. Frame transformation: R(theta0) * [x_local, y_local]
    rot_matrix = np.array([
        [np.cos(theta0), -np.sin(theta0)],
        [np.sin(theta0),  np.cos(theta0)]
    ])

    local_pos = np.array([x_local, y_local])
    global_pos = np.dot(rot_matrix, local_pos)

    dx_global = x0 + global_pos[0]
    dy_global = y0 + global_pos[1]

    # Trajectory for plotting
    trajectory_global = np.dot(rot_matrix, np.array(trajectory_local).T).T
    trajectory_global[:, 0] += x0
    trajectory_global[:, 1] += y0

    # Output: [dx_global, dy_global, delta_theta]
    return np.array([dx_global, dy_global, theta_local]), trajectory_global

# ==========================================
# 2. Main Execution and Evaluation
# ==========================================
def main():
    # --- A. Load Configuration ---
    with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)

    m = config['model']['physics']['mass']
    s = config['model']['physics']['size']
    angular_scale = config['model']['physics']['angular_scale']
    asym_sin1 = config['model']['physics']['asym_sin1']
    asym_cos1 = config['model']['physics']['asym_cos1']
    asym_sin2 = config['model']['physics']['asym_sin2']
    asym_cos2 = config['model']['physics']['asym_cos2']

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
    T = config['model']['physics']['push_duration']
    steps = config['model']['physics']['simulation_steps']
    dt = T / steps

    predictions = []
    trajectories = []
    starts = []

    print(f"Running physics simulation for {len(data_x)} pushes...")

    # --- D. Run the Simulation ---
    for i in range(len(data_x)):
        theta0 = data_x[i][0]
        d = data_x[i][1]       # contact point offset (side)
        D = data_x[i][2]       # total push distance

        s_start = [0.0, 0.0, theta0]
        starts.append(s_start)

        pred_state, traj = predict_physics(
            s_start, T, d, D, m, s, dt=dt,
            angular_scale=angular_scale,
            asym_sin1=asym_sin1, asym_cos1=asym_cos1,
            asym_sin2=asym_sin2, asym_cos2=asym_cos2)
        predictions.append(pred_state)
        trajectories.append(traj)

    predictions = np.array(predictions)

    # --- E. Calculate MSE Loss ---
    squared_errors = np.linalg.norm(predictions - data_y, axis=1)**2
    mse_loss = np.mean(squared_errors)

    print("-" * 40)
    print(f"Physics Model MSE Loss: {mse_loss:.6f}")
    print("-" * 40)

    # --- F. Plot Results ---
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
        ax.axis('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
