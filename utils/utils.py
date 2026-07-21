import random
import numpy as np
import torch

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