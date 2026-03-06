import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_model(model, optimizer, train_loader, num_epochs, device, val_data=None,
                scheduler=None, patience=None):
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    pbar = tqdm(range(num_epochs), desc=f"Training {model.__class__.__name__}")

    for epoch in pbar:
        model.train()
        total_loss = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

        avg_train = total_loss / total_samples
        train_losses.append(avg_train)

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                x_val, y_val = val_data
                val_pred = model(x_val)
                val_loss = F.mse_loss(val_pred, y_val).item()
            val_losses.append(val_loss)
            pbar.set_postfix(train=f"{avg_train:.6f}", val=f"{val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    pbar.close()
                    print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                    break
        else:
            pbar.set_postfix(loss=f"{avg_train:.6f}")

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val loss: {best_val_loss:.6f})")

    return train_losses, val_losses


def finetune_lbfgs(model, x_train, y_train, device, val_data=None,
                   max_iter=200, lr=1.0):
    x_train_t = torch.FloatTensor(x_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=20,
        line_search_fn="strong_wolfe"
    )

    best_val_mse = float("inf")
    best_state = None

    pbar = tqdm(range(max_iter), desc=f"LBFGS {model.__class__.__name__}")
    for step in pbar:
        def closure():
            optimizer.zero_grad()
            pred = model(x_train_t)
            loss = F.mse_loss(pred, y_train_t)
            loss.backward()
            return loss

        model.train()
        loss = optimizer.step(closure)
        train_mse = loss.item()

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                x_val, y_val = val_data
                val_pred = model(x_val)
                val_mse = F.mse_loss(val_pred, y_val).item()

            pbar.set_postfix(train=f"{train_mse:.6f}", val=f"{val_mse:.6f}")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = copy.deepcopy(model.state_dict())
        else:
            pbar.set_postfix(train=f"{train_mse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  LBFGS best val MSE: {best_val_mse:.6f}")

    return best_val_mse
