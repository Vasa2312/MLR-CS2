import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib.models import NNModel, NNPhysicsModel, FeatureTransform
from lib.physics import PushPhysics
from helpers.utils import load_data, prepare_dataloader
from helpers.config import load_config
from train import train_model, finetune_lbfgs
from colorama import init, Fore, Style

init()


def print_header(text: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def print_success(text: str):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")


def print_info(text: str):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train push planning model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()


def compute_norm_stats(x_train, y_train):
    feat_transform = FeatureTransform()
    with torch.no_grad():
        x_feat = feat_transform(torch.FloatTensor(x_train)).numpy()
    feat_mean = x_feat.mean(axis=0)
    feat_std = x_feat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    return dict(feat_mean=feat_mean, feat_std=feat_std, y_mean=y_mean, y_std=y_std)


def make_warmup_cosine_scheduler(optimizer, num_epochs, warmup_epochs=50):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = config.get_device()
    print_info(f"Using device: {device}")

    print_header("Loading Data")
    x_data, y_data = load_data(config)
    print_info(f"Loaded: x={x_data.shape}, y={y_data.shape}")

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
    patience = config.training.get("early_stopping_patience", 250)
    weight_decay = config.model["optimizer"].get("weight_decay", 1e-4)

    print_header("Computing Normalization Statistics")
    norm_stats = compute_norm_stats(x_train, y_train)
    y_std = y_train.std(axis=0)
    print_info(f"Output stds: dx={y_std[0]:.4f}, dy={y_std[1]:.4f}, dtheta={y_std[2]:.4f}")

    print_header("Evaluating Physics Model")
    physics = PushPhysics.from_config(config.model["physics"])
    with torch.no_grad():
        physics_pred = physics.compute_motion(x_val_t)
    physics_mse = torch.mean((physics_pred - y_val_t) ** 2).item()
    print_info(f"Physics Model Val MSE: {physics_mse:.6f}")

    print_info("Caching physics predictions...")
    with torch.no_grad():
        phys_train_t = physics.compute_motion(torch.FloatTensor(x_train).to(device))
        phys_train_np = phys_train_t.cpu().numpy()
        phys_val_np = physics_pred.cpu().numpy()

    x_train_aug = np.concatenate([x_train, phys_train_np], axis=1)
    x_val_aug = np.concatenate([x_val, phys_val_np], axis=1)
    x_val_aug_t = torch.FloatTensor(x_val_aug).to(device)
    val_data_aug = (x_val_aug_t, y_val_t)
    hybrid_train_loader = prepare_dataloader(x_train_aug, y_train, config)

    print_header("Training Neural Network")
    torch.manual_seed(42)
    np.random.seed(42)
    nn_model = NNModel(
        input_dim=net_cfg["input_dim"], output_dim=net_cfg["task_dim"],
        hidden_dims=net_cfg["hidden_dims"], **norm_stats,
    ).to(device)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, num_epochs, warmup_epochs=50)

    nn_train_losses, nn_val_losses = train_model(
        nn_model, optimizer, train_loader, num_epochs, device, val_data,
        scheduler=scheduler, patience=patience
    )

    print_info("LBFGS fine-tuning NN...")
    finetune_lbfgs(nn_model, x_train, y_train, device, val_data, max_iter=1000, lr=0.1)

    with torch.no_grad():
        nn_pred = nn_model(x_val_t)
        nn_mse = torch.mean((nn_pred - y_val_t) ** 2).item()
    print_success(f"NN Val MSE: {nn_mse:.6f}")

    print_header("Training Hybrid Model")
    torch.manual_seed(42)
    np.random.seed(42)
    hybrid_model = NNPhysicsModel(
        input_dim=net_cfg["input_dim"], output_dim=net_cfg["task_dim"],
        hidden_dims=net_cfg["hidden_dims"], physics=physics, **norm_stats,
    ).to(device)

    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, num_epochs, warmup_epochs=50)

    hybrid_train_losses, hybrid_val_losses = train_model(
        hybrid_model, optimizer, hybrid_train_loader, num_epochs, device, val_data_aug,
        scheduler=scheduler, patience=patience
    )

    print_info("LBFGS fine-tuning Hybrid...")
    finetune_lbfgs(hybrid_model, x_train_aug, y_train, device, val_data_aug, max_iter=1000, lr=0.1)

    with torch.no_grad():
        hybrid_pred = hybrid_model(x_val_aug_t)
        hybrid_mse = torch.mean((hybrid_pred - y_val_t) ** 2).item()
    print_success(f"Hybrid Val MSE: {hybrid_mse:.6f}")

    print_header("Results Summary")
    print_info(f"Physics Model Val MSE:  {physics_mse:.6f}")
    print_info(f"NN Model Val MSE:       {nn_mse:.6f}")
    print_info(f"Hybrid Model Val MSE:   {hybrid_mse:.6f}")
    best_mse = min(nn_mse, hybrid_mse)
    print_success(f"\nBest Val MSE: {best_mse:.6f}")

    os.makedirs("results", exist_ok=True)

    nn_pred_np = nn_pred.cpu().numpy()
    hybrid_pred_np = hybrid_pred.cpu().numpy()
    physics_pred_np = physics_pred.cpu().numpy()

    print_header("Per-Dimension MSE")
    for name, pred in [("Physics", physics_pred_np), ("NN", nn_pred_np), ("Hybrid", hybrid_pred_np)]:
        dim_mse = np.mean((pred - y_val) ** 2, axis=0)
        print_info(f"  {name:8s}  dx={dim_mse[0]:.6f}  dy={dim_mse[1]:.6f}  dtheta={dim_mse[2]:.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(nn_train_losses, label="NN Train")
    ax1.plot(hybrid_train_losses, label="Hybrid Train")
    ax1.axhline(y=physics_mse, color="r", linestyle="--", label="Physics")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss"); ax1.set_title("Training Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(nn_val_losses, label="NN Val")
    ax2.plot(hybrid_val_losses, label="Hybrid Val")
    ax2.axhline(y=physics_mse, color="r", linestyle="--", label="Physics")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss"); ax2.set_title("Validation Loss")
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print_success("Saved results/training_curves.png")

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
                         label=f"Gap: {gap[-1]:.5f}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title(f"{name}: Overfitting Check"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/overfitting_check.png", dpi=150)
    plt.close()
    print_success("Saved results/overfitting_check.png")

    dim_labels = ["dx (X)", "dy (Y)", "dθ (Theta)"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(3)
    width = 0.25
    for i, (name, pred_np) in enumerate([
        ("Physics", physics_pred_np), ("NN", nn_pred_np), ("Hybrid", hybrid_pred_np),
    ]):
        dim_mse = np.mean((pred_np - y_val) ** 2, axis=0)
        bars = ax.bar(x_pos + i * width, dim_mse, width, label=name)
        for bar, val in zip(bars, dim_mse):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.5f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel("MSE"); ax.set_title("Per-Dimension MSE"); ax.legend(); ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("results/per_dim_mse.png", dpi=150)
    plt.close()
    print_success("Saved results/per_dim_mse.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, pred_np) in enumerate([
        ("Physics", physics_pred_np), ("NN", nn_pred_np), ("Hybrid", hybrid_pred_np),
    ]):
        ax = axes[idx]
        ax.scatter(y_val[:, 0], y_val[:, 1], c="green", label="Ground Truth", alpha=0.5, s=10)
        ax.scatter(pred_np[:, 0], pred_np[:, 1], c="red", label="Predicted", alpha=0.5, s=10)
        ax.set_title(f"{name} Model"); ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/predictions.png", dpi=150)
    plt.close()
    print_success("Saved results/predictions.png")

    n_pushes = 10
    arrow_len = 0.015

    for name, pred_np in [
        ("Physics", physics_pred_np), ("NN", nn_pred_np), ("Hybrid", hybrid_pred_np),
    ]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        gt_x = np.cumsum(np.concatenate([[0], y_val[:n_pushes, 0]]))
        gt_y = np.cumsum(np.concatenate([[0], y_val[:n_pushes, 1]]))
        gt_theta = np.cumsum(np.concatenate([[0], y_val[:n_pushes, 2]]))

        pr_x = np.cumsum(np.concatenate([[0], pred_np[:n_pushes, 0]]))
        pr_y = np.cumsum(np.concatenate([[0], pred_np[:n_pushes, 1]]))
        pr_theta = np.cumsum(np.concatenate([[0], pred_np[:n_pushes, 2]]))

        ax1.plot(gt_x, gt_y, 'g-o', label="Ground Truth", markersize=8, linewidth=2)
        ax1.plot(pr_x, pr_y, 'r--s', label="Predicted", markersize=8, linewidth=2)
        ax1.plot(0, 0, 'k*', markersize=15, label="Start")

        for i in range(n_pushes + 1):
            ax1.arrow(gt_x[i], gt_y[i],
                      arrow_len * np.cos(gt_theta[i]), arrow_len * np.sin(gt_theta[i]),
                      head_width=0.004, head_length=0.002, fc='green', ec='green')
            ax1.arrow(pr_x[i], pr_y[i],
                      arrow_len * np.cos(pr_theta[i]), arrow_len * np.sin(pr_theta[i]),
                      head_width=0.004, head_length=0.002, fc='red', ec='red')
            ax1.annotate(str(i), (gt_x[i], gt_y[i]), textcoords="offset points",
                         xytext=(5, 5), fontsize=8, color="green")

        ax1.set_xlabel("X Position"); ax1.set_ylabel("Y Position")
        ax1.set_title(f"{name}: X-Y Trajectory ({n_pushes} pushes)")
        ax1.legend(); ax1.grid(True); ax1.axis("equal")

        pushes = np.arange(n_pushes + 1)
        ax2.plot(pushes, np.degrees(gt_theta), 'g-o', label="Ground Truth θ", markersize=8, linewidth=2)
        ax2.plot(pushes, np.degrees(pr_theta), 'r--s', label="Predicted θ", markersize=8, linewidth=2)
        ax2.set_xlabel("Push Number"); ax2.set_ylabel("Cumulative θ (degrees)")
        ax2.set_title(f"{name}: Orientation Over Pushes")
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(f"results/{safe_name}_trajectory.png", dpi=150)
        plt.close()
        print_success(f"Saved results/{safe_name}_trajectory.png")


if __name__ == "__main__":
    main()
