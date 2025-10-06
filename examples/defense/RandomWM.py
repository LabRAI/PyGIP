from pygip.datasets import Cora
from pygip.models.defense import RandomWM


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)

def randomwm():
    dataset = Cora(api_type='dgl')
    med = RandomWM(dataset, defense_ratio=0.1, wm_node=50, pr=0.1, pg=0.1)
    med.defend()


if __name__ == '__main__':
    randomwm()
