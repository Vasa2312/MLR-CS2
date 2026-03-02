import copy
import torch
from tqdm import tqdm


def train_model(model, optimizer, train_loader, num_epochs, device, val_data=None,
                scheduler=None):
    """
    Train a model, track best val weights, and return per-epoch losses.

    Args:
        model: nn.Module with a loss(pred, target) method
        optimizer: torch optimizer
        train_loader: DataLoader for training data
        num_epochs: number of epochs
        device: torch device
        val_data: optional (x_val, y_val) tensors already on device
        scheduler: optional LR scheduler

    Returns:
        train_losses: list of avg training loss per epoch
        val_losses: list of avg validation loss per epoch (empty if no val_data)
    """
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None

    pbar = tqdm(range(num_epochs), desc=f"Training {model.__class__.__name__}")

    for epoch in pbar:
        # Training
        model.train()
        total_loss = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = model.loss(pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

        avg_train = total_loss / total_samples
        train_losses.append(avg_train)

        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                x_val, y_val = val_data
                val_pred = model(x_val)
                val_loss = model.loss(val_pred, y_val).item()
            val_losses.append(val_loss)
            pbar.set_postfix(train=f"{avg_train:.6f}", val=f"{val_loss:.6f}")

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
        else:
            pbar.set_postfix(loss=f"{avg_train:.6f}")

        if scheduler is not None:
            scheduler.step()

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val MSE: {best_val_loss:.6f})")

    return train_losses, val_losses
