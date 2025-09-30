from pygip.datasets import *
from pygip.models.defense import GroveDefense


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (verification accuracy, defense time, inference time)

def grovedefense():
    dataset = Cora(api_type='dgl')
    med = GroveDefense(dataset, attack_node_fraction=0.1,
                       hidden_dim=256,
                       verification_threshold=0.5,
                       num_surrogate_models=3)
    _, res_comp = med.defend()


if __name__ == '__main__':
    grovedefense()
