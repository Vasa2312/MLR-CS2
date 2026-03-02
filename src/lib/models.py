import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

from .physics import PushPhysics


class ResBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class NNModel(nn.Module):
    """Neural network with BatchNorm and skip connections"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim == prev_dim:
                layers.append(ResBlock(hidden_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ])
                prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self._dim_weights = None

    def set_dim_weights(self, y_std: np.ndarray):
        """Set dimension weights for balanced loss (inverse variance weighting)"""
        weights = 1.0 / (y_std ** 2)
        weights = weights / weights.mean()
        self._dim_weights = torch.FloatTensor(weights)

    def forward(self, x):
        return self.layers(x)

    def loss(self, pred, target):
        if self._dim_weights is not None:
            w = self._dim_weights.to(pred.device)
            return torch.mean(((pred - target) ** 2) * w)
        return nn.functional.mse_loss(pred, target)


class NNPhysicsModel(NNModel):
    """Hybrid model: physics prediction + learned residual correction"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        physics: PushPhysics,
    ):
        super().__init__(input_dim + output_dim, output_dim, hidden_dims)
        self.physics = physics

    def forward(self, x):
        physics_pred = self.physics.compute_motion(x).detach()
        combined = torch.cat([x, physics_pred], dim=1)
        correction = self.layers(combined)
        return physics_pred + correction


class PushPlanner:
    """High-level push planning and training"""

    def __init__(
        self, model_config: Dict[str, Any], physics_sampling_config: Dict[str, Any]
    ):
        self.model_config = model_config
        self.physics_sampling_config = physics_sampling_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.forward_model = PushNetFactory.create(model_config)
        self.forward_model = self.forward_model.to(self.device)

        lr = model_config["optimizer"]["learning_rate"]
        self.forward_optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=lr
        )

    def train_epoch(self, dataloader):
        self.forward_model.train()
        total_loss = 0
        total_samples = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.forward_optimizer.zero_grad()
            pred = self.forward_model(x_batch)
            loss = self.forward_model.loss(pred, y_batch)
            loss.backward()
            self.forward_optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

        return total_loss / total_samples


class PushNetFactory:
    """Factory for creating different types of push networks"""

    @staticmethod
    def create(config: Dict[str, Any]) -> nn.Module:
        network_config = config["network"]
        physics_config = config["physics"]
        model_type = network_config["type"]
        hidden_dims = network_config["hidden_dims"]

        if model_type == "NNModel":
            return NNModel(
                network_config["input_dim"], network_config["task_dim"], hidden_dims
            )
        elif model_type == "PhysicsModel":
            return PushPhysics.from_config(physics_config)
        else:
            physics = PushPhysics.from_config(physics_config)
            return NNPhysicsModel(
                network_config["input_dim"],
                network_config["task_dim"],
                hidden_dims,
                physics,
            )
