import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

from .physics import PushPhysics


class NNModel(nn.Module):
    """Base neural network architecture"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    # TODO: Implement forward function
    # TODO: Implement loss function
    # TODO: Implement accuracy function


class NNPhysicsModel(BaseNet):
    """Neural network with physics knowledge"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        physics: PushPhysics,
    ):
        super().__init__(input_dim + output_dim, output_dim, hidden_dims)
        self.physics = physics
        self.requires_grad = True

    # TODO: Implement forward function


class PushPlanner:
    """High-level push planning and training"""

    def __init__(
        self, model_config: Dict[str, Any], physics_sampling_config: Dict[str, Any]
    ):
        self.model_config = model_config
        self.physics_sampling_config = physics_sampling_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: Initialize models
        # TODO: Move models to device
        # TODO: Setup optimizers

    # TODO: Implement optimize_push function
    # TODO: Implement plan_push function
    # TODO: Implement train_epoch function


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

    # TODO: Expand factory as needed
