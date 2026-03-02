import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib.models import NNModel, NNPhysicsModel
from lib.physics import PushPhysics
from helpers.utils import load_data, prepare_dataloader
from helpers.config import load_config
from train import train_model
from colorama import init, Fore, Style

# Initialize colorama
init()


def print_header(text: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def print_success(text: str):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")


def print_info(text: str):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")


def print_error(text: str):
    print(f"{Fore.RED}{text}{Style.RESET_ALL}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train push planning model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    device = config.get_device()
    print_info(f"Using device: {device}")

    # Load data
    print_header("Loading Data")
    x_data, y_data = load_data(config)
    print_info(f"Loaded data shapes: x={x_data.shape}, y={y_data.shape}")

    # Train/val split (80/20)
    n = len(x_data)
    split = int(0.8 * n)
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:split], indices[split:]

    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]

    train_loader = prepare_dataloader(x_train, y_train, config)

    x_val_t = torch.FloatTensor(x_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    val_data = (x_val_t, y_val_t)

    net_cfg = config.model["network"]
    lr = config.model["optimizer"]["learning_rate"]
    num_epochs = config.training["num_epochs"]

    # ========== 1. Physics Model ==========
    print_header("Evaluating Physics Model")
    physics = PushPhysics.from_config(config.model["physics"])
    with torch.no_grad():
        physics_pred = physics.compute_motion(x_val_t)
    physics_mse = torch.mean((physics_pred - y_val_t) ** 2).item()
    print_info(f"Physics Model Val MSE: {physics_mse:.6f}")

    # Compute dimension weights for balanced loss
    y_std = y_train.std(axis=0)
    print_info(f"Output stds: dx={y_std[0]:.4f}, dy={y_std[1]:.4f}, dtheta={y_std[2]:.4f}")

    # ========== 2. Neural Network Model ==========
    print_header("Training Neural Network Model")
    nn_model = NNModel(
        net_cfg["input_dim"], net_cfg["task_dim"], net_cfg["hidden_dims"]
    ).to(device)
    nn_model.set_dim_weights(y_std)
    nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=1e-5)
    nn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(nn_optimizer, T_max=num_epochs)
    nn_train_losses, nn_val_losses = train_model(
        nn_model, nn_optimizer, train_loader, num_epochs, device, val_data,
        scheduler=nn_scheduler
    )
    nn_best_val = min(nn_val_losses)
    print_info(f"NN Model Best Val MSE: {nn_best_val:.6f}")

    # ========== 3. Hybrid (NN + Physics) Model ==========
    print_header("Training Hybrid (NN + Physics) Model")
    hybrid_model = NNPhysicsModel(
        net_cfg["input_dim"], net_cfg["task_dim"], net_cfg["hidden_dims"], physics
    ).to(device)
    hybrid_model.set_dim_weights(y_std)
    hybrid_optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=lr, weight_decay=1e-5)
    hybrid_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hybrid_optimizer, T_max=num_epochs)
    hybrid_train_losses, hybrid_val_losses = train_model(
        hybrid_model, hybrid_optimizer, train_loader, num_epochs, device, val_data,
        scheduler=hybrid_scheduler
    )
    hybrid_best_val = min(hybrid_val_losses)
    print_info(f"Hybrid Model Best Val MSE: {hybrid_best_val:.6f}")

    # ========== Results Summary ==========
    print_header("Results Summary")
    print_info(f"Physics Model Val MSE:  {physics_mse:.6f}")
    print_info(f"NN Model Val MSE:       {nn_best_val:.6f}")
    print_info(f"Hybrid Model Val MSE:   {hybrid_best_val:.6f}")
    print_success("\nTraining completed!")

    # ========== Save Plots ==========
    os.makedirs("results", exist_ok=True)

    # Get predictions (best model weights restored by early stopping)
    with torch.no_grad():
        nn_pred = nn_model(x_val_t).cpu().numpy()
        hybrid_pred_np = hybrid_model(x_val_t).cpu().numpy()
    physics_pred_np = physics_pred.cpu().numpy()

    # --- Plot 1: Training curves (train vs val) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(nn_train_losses, label="NN Train")
    ax1.plot(hybrid_train_losses, label="Hybrid Train")
    ax1.axhline(y=physics_mse, color="r", linestyle="--", label="Physics (no training)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(nn_val_losses, label="NN Val")
    ax2.plot(hybrid_val_losses, label="Hybrid Val")
    ax2.axhline(y=physics_mse, color="r", linestyle="--", label="Physics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print_success("Saved results/training_curves.png")

    # --- Plot 2: Overfitting diagnostic (train vs val gap) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name, t_losses, v_losses in [
        (axes[0], "NN", nn_train_losses, nn_val_losses),
        (axes[1], "Hybrid", hybrid_train_losses, hybrid_val_losses),
    ]:
        epochs = range(1, len(t_losses) + 1)
        ax.plot(epochs, t_losses, label="Train Loss", color="blue")
        ax.plot(epochs, v_losses, label="Val Loss", color="orange")
        gap = [v - t for t, v in zip(t_losses, v_losses)]
        ax.fill_between(epochs, t_losses, v_losses, alpha=0.2,
                         color="red" if gap[-1] > 0 else "green",
                         label=f"Gap (val-train): {gap[-1]:.5f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{name} Model: Overfitting Check")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/overfitting_check.png", dpi=150)
    plt.close()
    print_success("Saved results/overfitting_check.png")

    # --- Plot 3: Per-dimension MSE breakdown ---
    dim_labels = ["dx (X)", "dy (Y)", "dθ (Theta)"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(3)
    width = 0.25

    for i, (name, pred_np) in enumerate([
        ("Physics", physics_pred_np),
        ("NN", nn_pred),
        ("Hybrid", hybrid_pred_np),
    ]):
        dim_mse = np.mean((pred_np - y_val) ** 2, axis=0)
        bars = ax.bar(x_pos + i * width, dim_mse, width, label=name)
        for bar, val in zip(bars, dim_mse):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.5f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel("MSE")
    ax.set_title("Per-Dimension MSE Breakdown")
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("results/per_dim_mse.png", dpi=150)
    plt.close()
    print_success("Saved results/per_dim_mse.png")

    # --- Plot 4: Predictions scatter ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, pred_np) in enumerate([
        ("Physics", physics_pred_np),
        ("NN", nn_pred),
        ("Hybrid", hybrid_pred_np),
    ]):
        ax = axes[idx]
        ax.scatter(y_val[:, 0], y_val[:, 1], c="green", label="Ground Truth", alpha=0.5, s=10)
        ax.scatter(pred_np[:, 0], pred_np[:, 1], c="red", label="Predicted", alpha=0.5, s=10)
        ax.set_title(f"{name} Model")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/predictions.png", dpi=150)
    plt.close()
    print_success("Saved results/predictions.png")

    # --- Plot 5: Individual push examples ---
    num_examples = min(3, len(y_val))
    for name, pred_np in [
        ("Physics", physics_pred_np),
        ("NN", nn_pred),
        ("Hybrid", hybrid_pred_np),
    ]:
        fig, axes = plt.subplots(1, num_examples, figsize=(15, 5))
        for i in range(num_examples):
            ax = axes[i]
            ax.scatter(0, 0, c="blue", marker="o", s=100, label="Start")
            ax.scatter(y_val[i, 0], y_val[i, 1], c="green", marker="s", s=100, label="Ground Truth Final")
            ax.scatter(pred_np[i, 0], pred_np[i, 1], c="red", marker="X", s=100, label="Predicted Final")
            ax.set_title(f"Push Example {i+1}")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True)
            ax.legend()
            ax.axis("equal")
        fig.suptitle(f"{name} Model", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"results/{name.lower()}_examples.png", dpi=150)
        plt.close()
        print_success(f"Saved results/{name.lower()}_examples.png")


if __name__ == "__main__":
    main()
