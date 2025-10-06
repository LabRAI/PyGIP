from pygip.datasets import *
from pygip.models.attack import ModelExtractionAttack0, ModelExtractionAttack1, ModelExtractionAttack2, \
    ModelExtractionAttack3, ModelExtractionAttack4, ModelExtractionAttack5
from pygip.utils.hardware import set_device

# TODO verify performance
# TODO generate shadow graph
# TODO attack after defense
# TODO record metrics (original acc, attack acc, fidelity)

set_device("cpu")  # cpu, cuda:0


def mea0():
    dataset = CiteSeer(api_type='dgl')
    print(dataset)
    mea = ModelExtractionAttack0(dataset, attack_node_fraction=0.1)
    mea.attack()


def mea1():
    dataset = Cora(api_type='dgl')
    mea = ModelExtractionAttack1(dataset, attack_node_fraction=0.1)
    mea.attack()


def mea2():
    dataset = Cora(api_type='dgl')
    mea = ModelExtractionAttack2(dataset, attack_node_fraction=0.1)
    mea.attack()


def mea3():
    dataset = Cora(api_type='dgl')
    mea = ModelExtractionAttack3(dataset, attack_node_fraction=0.1)
    mea.attack()


def mea4():
    dataset = Cora(api_type='dgl')
    mea = ModelExtractionAttack4(dataset, attack_node_fraction=0.1)
    mea.attack()


def mea5():
    dataset = Cora(api_type='dgl')
    mea = ModelExtractionAttack5(dataset, attack_node_fraction=0.1)
    mea.attack()


if __name__ == '__main__':
    mea0()
