import torch
import numpy as np
from typing import Dict, Any


class PushPhysics:
    """Physics engine for push interactions with asymmetry correction"""

    def __init__(
        self, mass: float = 0.1, size: float = 0.1, inertia_factor: float = 1 / 12,
        angular_scale: float = 77.12,
        asym_sin1: float = -2.00, asym_cos1: float = 0.56,
        asym_sin2: float = -1.07, asym_cos2: float = 0.24,
    ):
        # Object properties
        self.mass = float(mass)
        self.size = float(size)
        self.inertia_factor = float(inertia_factor)
        self.inertia = self.inertia_factor * self.mass * (self.size**2)

        # Angular dynamics parameters
        self.angular_scale = float(angular_scale)
        self.asym_sin1 = float(asym_sin1)
        self.asym_cos1 = float(asym_cos1)
        self.asym_sin2 = float(asym_sin2)
        self.asym_cos2 = float(asym_cos2)

        # Default simulation parameters
        self._push_duration = 3.0
        self._simulation_steps = 300

    @classmethod
    def from_config(cls, physics_config: Dict[str, Any]) -> "PushPhysics":
        """Create PushPhysics instance from config dictionary"""
        instance = cls(
            mass=physics_config.get("mass", 0.1),
            size=physics_config.get("size", 0.1),
            inertia_factor=physics_config.get("inertia_factor", 1 / 12),
            angular_scale=physics_config.get("angular_scale", 77.12),
            asym_sin1=physics_config.get("asym_sin1", -2.00),
            asym_cos1=physics_config.get("asym_cos1", 0.56),
            asym_sin2=physics_config.get("asym_sin2", -1.07),
            asym_cos2=physics_config.get("asym_cos2", 0.24),
        )
        instance._push_duration = physics_config.get("push_duration", 3.0)
        instance._simulation_steps = physics_config.get("simulation_steps", 300)
        return instance

    def compute_motion(
        self, push_params: torch.Tensor, duration: float = None, steps: int = None
    ) -> torch.Tensor:
        """
        Compute object motion using clarification equations with asymmetry correction.

        The angular rate accounts for both the offset-driven rotation (o * V)
        and the object's asymmetric geometry (theta0-dependent correction).

        Args:
            push_params: [batch_size, 3] tensor of [rotation, side, distance]
            duration: Duration of push in seconds (optional)
            steps: Number of simulation steps (optional)

        Returns:
            [batch_size, 3] tensor of [x, y, theta] final states
        """
        T = duration if duration is not None else self._push_duration
        N = steps if steps is not None else self._simulation_steps
        dt = T / N

        # Extract push parameters
        theta0 = push_params[:, 0]   # initial orientation
        d = push_params[:, 1]        # contact point offset (o in clarification)
        D = push_params[:, 2]        # total push distance

        # Velocity profile: v_max = 2D/T
        v_max = 2.0 * D / T

        # Initialize local frame states
        x_local = torch.zeros_like(theta0)
        y_local = torch.zeros_like(theta0)
        theta_local = torch.zeros_like(theta0)

        # Asymmetry correction: accounts for non-square object geometry
        # causing push-direction-dependent rotation even when offset=0
        asym_rate = (
            self.asym_sin1 * torch.sin(theta0)
            + self.asym_cos1 * torch.cos(theta0)
            + self.asym_sin2 * torch.sin(2 * theta0)
            + self.asym_cos2 * torch.cos(2 * theta0)
        )

        # Numerical integration (clarification equations)
        for i in range(N):
            t_i = i * dt
            # 1. Velocity: v(t) = v_max * (0.5*sin(2*pi*t/T - pi/2) + 0.5)
            v_i = (v_max / 2.0) * (np.sin(2.0 * np.pi * t_i / T - np.pi / 2.0) + 1.0)

            # 2. Angular update: dtheta = (K*o + asymmetry) * v * dt
            theta_local = theta_local + (self.angular_scale * d + asym_rate) * v_i * dt

            # 3. Position update (local frame): dx = -v*cos(theta)*dt
            x_local = x_local - v_i * torch.cos(theta_local) * dt
            y_local = y_local - v_i * torch.sin(theta_local) * dt

        # 4. Frame transformation: R(theta0) * [x_local, y_local]
        cos_t = torch.cos(theta0)
        sin_t = torch.sin(theta0)
        x_global = cos_t * x_local - sin_t * y_local
        y_global = sin_t * x_local + cos_t * y_local

        # Output deltas: [dx_global, dy_global, delta_theta]
        return torch.stack([x_global, y_global, theta_local], dim=1)
