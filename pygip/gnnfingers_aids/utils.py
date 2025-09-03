
import random, numpy as np, torch

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AvgMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.s=0.0; self.n=0
    def add(self, v, k=1):
        self.s += float(v)*k; self.n += k
    @property
    def avg(self):
        return self.s/max(1,self.n)
