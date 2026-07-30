import random
import pathlib
import numpy as np
import torch


def increment_path(base_dir="runs/exp"):
    """Returns the next unused incremented experiment directory (runs/exp0, runs/exp1, ...),
    YOLO-style, and creates it."""
    base_dir = pathlib.Path(base_dir)
    parent   = base_dir.parent
    stem     = base_dir.name
    parent.mkdir(parents=True, exist_ok=True)

    i = 0
    while (parent / f"{stem}{i}").exists():
        i += 1
    exp_dir = parent / f"{stem}{i}"
    exp_dir.mkdir(parents=True)
    print(f"Experiment results will be saved to {exp_dir}")
    return exp_dir


def set_random_seeds(seed=42):
    """Locks all random seeds for 100% reproducible training and evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Random seeds locked to {seed}.")