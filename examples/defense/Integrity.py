from pygip.datasets import Cora
from pygip.models.defense import IntegrityVerification


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)

def integrity():
    dataset = Cora(api_type='dgl')
    med = IntegrityVerification(dataset, defense_ratio=0.1)
    med.defend()


if __name__ == '__main__':
    integrity()
