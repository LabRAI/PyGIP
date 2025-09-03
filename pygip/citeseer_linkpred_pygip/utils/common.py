# utils/common.py
import random
import numpy as np
import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
