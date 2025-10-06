from pygip.datasets import *
from pygip.models.attack import CEGA


# TODO verify performance
# TODO record metrics (original acc, attack acc, fidelity)


def cega():
    dataset = Cora(api_type='dgl')
    mea = CEGA(dataset, attack_node_fraction=0.1)
    mea.attack()


if __name__ == '__main__':
    cega()
