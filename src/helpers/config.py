import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass
import torch


@dataclass
class Config:
    data: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    device: Dict[str, Any]
    physics_sampling: Dict[str, Any]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

            required_sections = [
                "data",
                "model",
                "training",
                "device",
                "physics_sampling",
            ]
            for section in required_sections:
                if section not in config_dict:
                    config_dict[section] = {}

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        return cls(**config_dict)

    def update(self, override_config: Dict[str, Any]) -> None:
        for key, value in override_config.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, dict) and isinstance(value, dict):
                    current.update(value)
                else:
                    setattr(self, key, value)

    def get_device(self) -> torch.device:
        if self.device.get("use_cuda", False) and torch.cuda.is_available():
            return torch.device(f"cuda:{self.device.get('cuda_device', 0)}")
        return torch.device("cpu")


def load_config(
    config_path: str = None, override_config: Dict[str, Any] = None
) -> Config:
    if config_path is None:
        config_path = os.path.join("config", "default.yaml")

    config = Config.from_yaml(config_path)

    if override_config:
        config.update(override_config)

    return config
