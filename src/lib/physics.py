import torch
import numpy as np
from typing import Dict, Any


class PushPhysics:
    """
    Physics engine for planar push interactions.

    Uses the appendix equations with proper rotational dynamics:
        tau = m * v(t) * d           (torque from applied force)
        alpha = tau / I              (angular acceleration)
        delta_theta = 0.5 * alpha * dt^2  (kinematic update)
        dx = -v(t) * cos(theta) * dt
        dy = -v(t) * sin(theta) * dt

    All quantities are dimensionally consistent:
        tau [N·m], alpha [rad/s²], delta_theta [rad], v [m/s], dt [s].
    """

    def __init__(
        self, mass: float = 0.1, size: float = 0.1, inertia_factor: float = 1 / 12,
    ):
        self.mass = float(mass)
        self.size = float(size)
        self.inertia_factor = float(inertia_factor)
        self.inertia = self.inertia_factor * self.mass * (self.size ** 2)

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
        )
        instance._push_duration = physics_config.get("push_duration", 3.0)
        instance._simulation_steps = physics_config.get("simulation_steps", 300)
        return instance

    def compute_motion(
        self, push_params: torch.Tensor, duration: float = None, steps: int = None
    ) -> torch.Tensor:
        """
        Compute object motion using proper rigid-body dynamics.

        Args:
            push_params: [batch_size, 3] tensor of [theta0, offset, distance]
            duration: Duration of push in seconds (optional)
            steps: Number of simulation steps (optional)

        Returns:
            [batch_size, 3] tensor of [dx_global, dy_global, delta_theta]
        """
        T = duration if duration is not None else self._push_duration
        N = steps if steps is not None else self._simulation_steps
        dt = T / N

        # Extract push parameters
        theta0 = push_params[:, 0]   # initial orientation
        d = push_params[:, 1]        # contact point offset
        D = push_params[:, 2]        # total push distance

        # Velocity profile: v_max = 2D/T
        v_max = 2.0 * D / T

        # Initialize local frame states
        x_local = torch.zeros_like(theta0)
        y_local = torch.zeros_like(theta0)
        theta_local = torch.zeros_like(theta0)

        I = self.inertia  # kg·m²

        # Numerical integration
        for i in range(N):
            t_i = i * dt
            # 1. Velocity profile
            v_i = (v_max / 2.0) * (np.sin(2.0 * np.pi * t_i / T - np.pi / 2.0) + 1.0)

            # 2. Angular update: tau = m*v*d, alpha = tau/I, dtheta = 0.5*alpha*dt²
            tau_i = self.mass * v_i * d
            alpha_i = tau_i / I
            delta_theta_i = 0.5 * alpha_i * (dt ** 2)
            theta_local = theta_local + delta_theta_i

            # 3. Position update
            x_local = x_local - v_i * torch.cos(theta_local) * dt
            y_local = y_local - v_i * torch.sin(theta_local) * dt

        # 4. Frame transformation: R(theta0) * [x_local, y_local]
        cos_t = torch.cos(theta0)
        sin_t = torch.sin(theta0)
        x_global = cos_t * x_local - sin_t * y_local
        y_global = sin_t * x_local + cos_t * y_local

        return torch.stack([x_global, y_global, theta_local], dim=1)
