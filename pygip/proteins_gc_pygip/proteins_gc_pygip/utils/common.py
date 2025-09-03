
import random
import numpy as np
import torch

SEED = 7

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AvgMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.s = 0.0
        self.n = 0
    def add(self, v, k=1):
        self.s += float(v) * k
        self.n += k
    @property
    def avg(self):
        return self.s / max(1, self.n)
