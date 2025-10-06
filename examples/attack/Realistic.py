from pygip.datasets import Cora
from pygip.models.attack import RealisticAttack
from pygip.utils.hardware import set_device

# TODO verify performance
# TODO attack after defense
# TODO record metrics (original acc, attack acc, fidelity)

set_device("cpu")


def realistic():
    dataset = Cora(api_type='dgl')
    mea = RealisticAttack(
        dataset=dataset,
        attack_node_fraction=0.05,
        hidden_dim=64,
        threshold_s=0.6,  # Cosine similarity threshold
        threshold_a=0.4  # Edge prediction threshold
    )
    mea.attack()


if __name__ == '__main__':
    realistic()
