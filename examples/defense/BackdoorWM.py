from pygip.datasets import *
from pygip.models.defense import BackdoorWM


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)

def backdoorwm():
    dataset = Cora(api_type='dgl')
    med = BackdoorWM(dataset, attack_node_fraction=0.1, trigger_rate=0.1, l=20, target_label=0)
    med.defend()


if __name__ == '__main__':
    backdoorwm()
