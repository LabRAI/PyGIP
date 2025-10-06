from pygip.datasets import *
from pygip.models.attack import AdvMEA
from pygip.utils.hardware import set_device

# TODO verify performance
# TODO attack after defense
# TODO record metrics (original acc, attack acc, fidelity)

set_device("cpu")  # cpu, cuda:0


def advmea():
    dataset = Photo(api_type='dgl')
    mea = AdvMEA(dataset, attack_node_fraction=0.1)
    mea.attack()


if __name__ == '__main__':
    advmea()
