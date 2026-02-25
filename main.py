import argparse
import torch
from lib.models import PushPlanner
from helpers.utils import (
    load_data,
    prepare_dataloader,
    evaluate_planner,
    save_checkpoint,
    load_checkpoint,
)
from helpers.config import load_config
from tqdm import tqdm
from colorama import init, Fore, Style

# Initialize colorama
init()


def print_header(text: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def print_success(text: str):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")


def print_info(text: str):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")


def print_error(text: str):
    print(f"{Fore.RED}{text}{Style.RESET_ALL}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train push planning model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    device = config.get_device()
    print_info(f"Using device: {device}")

    # Load data
    print_header("Loading Data")
    x_data, y_data = load_data(config)
    print_info(f"Loaded data shapes: x={x_data.shape}, y={y_data.shape}")

    # ToDO: Call Dataloader

    # ToDO: Call Physics Push Planner

    print_header("Starting Training")
    pbar = tqdm(range(config.training["num_epochs"]), desc="Training Progress")

    # ToDO: Implement training loop
    for epoch in pbar:
        ...

    print_success("\nTraining completed!")

    # Test prediction
    print_header("Testing Prediction")

    # ToDo: Test the Model

    # ToDO: Make Some predictions with model
    # Make predictions with the model
    with torch.no_grad():
        # Move tensors to appropriate device
        # Do forward and calculate error


if __name__ == "__main__":
    main()
