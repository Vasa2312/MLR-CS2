import torch
import torch.nn as nn
from typing import List

from .physics import PushPhysics

FEATURE_DIM = 20  # output dimension of FeatureTransform


class FeatureTransform(nn.Module):
    """Expand raw [theta0, offset, distance] -> 20 engineered features."""

    def forward(self, x):
        theta0 = x[:, 0:1]
        offset = x[:, 1:2]
        distance = x[:, 2:3]
        s1, c1 = torch.sin(theta0), torch.cos(theta0)
        s2, c2 = torch.sin(2 * theta0), torch.cos(2 * theta0)
        return torch.cat([
            x,                                          # 3: raw
            s1, c1, s2, c2,                             # 4: harmonics 1-2
            torch.sin(3 * theta0), torch.cos(3 * theta0),  # 2: harmonic 3
            torch.sin(4 * theta0), torch.cos(4 * theta0),  # 2: harmonic 4
            offset ** 2, distance ** 2,                 # 2: quadratic
            offset * distance,                          # 1: cross
            offset * s1, offset * c1,                   # 2: offset-angle
            distance * s1, distance * c1,               # 2: distance-angle
            offset * distance * s1,                     # 1: triple interaction
            offset * distance * c1,                     # 1: triple interaction
        ], dim=1)  # total: 20


class NNModel(nn.Module):
    """Simple MLP with feature engineering and normalization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 feat_mean=None, feat_std=None, y_mean=None, y_std=None):
        super().__init__()
        self.feature_transform = FeatureTransform()
        self.output_dim = output_dim

        # Normalization buffers
        self.register_buffer('feat_mean',
            torch.zeros(FEATURE_DIM) if feat_mean is None else torch.FloatTensor(feat_mean))
        self.register_buffer('feat_std',
            torch.ones(FEATURE_DIM) if feat_std is None else torch.FloatTensor(feat_std))
        self.register_buffer('y_mean',
            torch.zeros(output_dim) if y_mean is None else torch.FloatTensor(y_mean))
        self.register_buffer('y_std',
            torch.ones(output_dim) if y_std is None else torch.FloatTensor(y_std))

        # Simple MLP: Linear -> ReLU -> ... -> Linear
        layers = []
        prev = FEATURE_DIM
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.feature_transform(x)
        feat_norm = (feat - self.feat_mean) / self.feat_std
        out_norm = self.layers(feat_norm)
        return out_norm * self.y_std + self.y_mean


class NNPhysicsModel(nn.Module):
    """Hybrid model: physics predictions as input features to NN."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 physics: PushPhysics,
                 feat_mean=None, feat_std=None, y_mean=None, y_std=None):
        super().__init__()
        self.feature_transform = FeatureTransform()
        self.physics = physics
        self.output_dim = output_dim

        # Normalization buffers
        self.register_buffer('feat_mean',
            torch.zeros(FEATURE_DIM) if feat_mean is None else torch.FloatTensor(feat_mean))
        self.register_buffer('feat_std',
            torch.ones(FEATURE_DIM) if feat_std is None else torch.FloatTensor(feat_std))
        self.register_buffer('y_mean',
            torch.zeros(output_dim) if y_mean is None else torch.FloatTensor(y_mean))
        self.register_buffer('y_std',
            torch.ones(output_dim) if y_std is None else torch.FloatTensor(y_std))

        # MLP input: 20 features + 3 physics outputs = 23
        layers = []
        prev = FEATURE_DIM + output_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Support augmented input: [input(3), cached_physics(3)]
        if x.shape[1] > 3:
            physics_pred = x[:, 3:6]
            x = x[:, :3]
        else:
            physics_pred = self.physics.compute_motion(x).detach()

        feat = self.feature_transform(x)
        feat_norm = (feat - self.feat_mean) / self.feat_std
        phys_norm = (physics_pred - self.y_mean) / self.y_std
        combined = torch.cat([feat_norm, phys_norm], dim=1)
        correction_norm = self.layers(combined)
        return physics_pred + correction_norm * self.y_std
