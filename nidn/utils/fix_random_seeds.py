import torch
import numpy as np


def fix_random_seeds(seed: int = 42):
    """This function sets the random seeds in torch and numpy to enable reproducible behavior.

    Args:
        seed (int): Seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
