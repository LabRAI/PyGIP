from pygip.datasets import *
from pygip.models.attack.EGSteal.EGSteal import EGSteal
from pygip.utils.hardware import set_device

set_device("cuda:0")  # cpu, cuda:0


def egsteal():
    dataset = MUTAGGraphClassification(api_type='pyg')
    egsteal = EGSteal(dataset,query_shadow_ratio=0.3)
    egsteal.attack()

if __name__ == '__main__':
    egsteal()
