import os
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader


def load_data(config) -> Tuple[np.ndarray, np.ndarray]:
    base_path = config.data["base_path"]
    x_path = os.path.join(base_path, config.data["train_x"])
    y_path = os.path.join(base_path, config.data["train_y"])

    x_data = np.load(x_path)
    y_data = np.load(y_path)
    return x_data, y_data


def prepare_dataloader(
    x_data: np.ndarray, y_data: np.ndarray, config
) -> torch.utils.data.DataLoader:
    x_tensor = torch.FloatTensor(x_data)
    y_tensor = torch.FloatTensor(y_data)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset, batch_size=config.data["batch_size"], shuffle=config.data["shuffle"]
    )


def evaluate_planner(
    planner, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, float]:
    planner.forward_model.eval()
    total_forward_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = planner.forward_model(x_batch)
            forward_loss = planner.forward_model.loss(y_pred, y_batch)

            batch_size = x_batch.size(0)
            total_forward_loss += forward_loss.item() * batch_size
            total_samples += batch_size

    avg_forward_loss = total_forward_loss / total_samples

    return avg_forward_loss


def save_checkpoint(
    planner, epoch: int, forward_loss: float, backward_loss: float, config
) -> None:
    os.makedirs(config.training["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(
        config.training["checkpoint_dir"], f"model_epoch_{epoch}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "forward_model_state_dict": planner.forward_model.state_dict(),
            "forward_optimizer_state_dict": planner.forward_optimizer.state_dict(),
            "forward_loss": forward_loss,
        },
        checkpoint_path,
    )


def load_checkpoint(planner, checkpoint_path: str) -> Tuple[int, float, float]:
    checkpoint = torch.load(checkpoint_path)
    planner.forward_model.load_state_dict(checkpoint["forward_model_state_dict"])
    planner.forward_optimizer.load_state_dict(
        checkpoint["forward_optimizer_state_dict"]
    )
    return checkpoint["epoch"], checkpoint["forward_loss"], checkpoint["backward_loss"]
