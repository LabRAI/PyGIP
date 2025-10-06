from pygip.datasets import Cora
from pygip.models.attack import DFEATypeI, DFEATypeII
from pygip.utils.hardware import set_device

# TODO verify performance
# TODO attack after defense
# TODO record metrics (original acc, attack acc, fidelity)

set_device("cpu")


def dfea_type1():
    dataset = Cora(api_type='dgl')
    mea = DFEATypeI(dataset, attack_node_fraction=0.1)
    mea.attack()


def dfea_type2():
    dataset = Cora(api_type='dgl')
    mea = DFEATypeII(dataset, attack_node_fraction=0.1)
    mea.attack()


if __name__ == '__main__':
    dfea_type2()
