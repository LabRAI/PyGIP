from pygip.datasets import Cora
from pygip.models.defense import ImperceptibleWM


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)

def imperceptiblewm():
    dataset = Cora(api_type='pyg')
    med = ImperceptibleWM(dataset, defense_ratio=0.1)
    med.defend()


if __name__ == '__main__':
    imperceptiblewm()
