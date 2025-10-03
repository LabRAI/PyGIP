from pygip.datasets import Cora
from pygip.models.defense import ATOM


# TODO test datasets
# TODO generate query set
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)

def atom():
    dataset = Cora(api_type='pyg')
    med = ATOM(dataset, attack_node_fraction=0.1)
    med.defend()


if __name__ == '__main__':
    atom()
