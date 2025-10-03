from pygip.datasets import Cora
from pygip.models.defense import SurviveWM


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)

def survivewm():
    dataset = Cora(api_type='dgl')
    med = SurviveWM(dataset, defense_ratio=0.1)
    med.defend()


if __name__ == '__main__':
    survivewm()
