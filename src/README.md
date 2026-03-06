# Push & Learn: Learning Dynamics Through Learning

RBE 577 - Machine Learning for Robotics | Project 2

## Overview

This project implements and compares three models for predicting the planar motion of a cracker box pushed by a UR10 robotic manipulator:

1. **Physics Model** - Rigid-body dynamics with numerical integration
2. **Neural Network Model** - MLP with engineered trigonometric features
3. **Hybrid Model** - Physics predictions as input features + learned residual correction

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full training pipeline:

```bash
cd src
python main.py
```

This trains all three models, evaluates them, and saves result plots to `results/`.

To use a custom config:

```bash
python main.py --config config/default.yaml
```

## Project Structure

```
src/
  config/default.yaml       # Hyperparameters and physics constants
  lib/models.py             # NNModel, NNPhysicsModel, FeatureTransform
  lib/physics.py            # PushPhysics engine
  helpers/utils.py          # Data loading, DataLoader preparation
  helpers/config.py         # YAML config loading, device management
  main.py                   # Full training pipeline and visualization
  train.py                  # train_model() and finetune_lbfgs()
  results/                  # Output plots
```

## Results

| Model | Total MSE | dx MSE | dy MSE | dtheta MSE | Improvement |
|---|---|---|---|---|---|
| Physics | 0.171645 | 0.006011 | 0.003219 | 0.505705 | --- |
| Neural Network | 0.000296 | 0.000024 | 0.000013 | 0.000851 | 580x |
| Hybrid (NN+Phys) | 0.000262 | 0.000132 | 0.000045 | 0.000609 | 655x |

## Dependencies

- Python 3.10+
- PyTorch 2.4.0+
- NumPy
- Matplotlib
- PyYAML
- colorama
