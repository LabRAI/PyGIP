from pygip.datasets import TUGraph
from pygip.models.attack import EGStealAttack

dataset = TUGraph(name="NCI109", api_type="pyg")

config = {
    "gnn_backbone": "GIN",
    "gnn_layers": 3,
    "hidden_dim": 128,
    "epochs": 5,          # set to 200 later
    "batch_size": 64,
    "explanation_mode": "CAM",
    "align_weight": 1.0,
}

attack = EGStealAttack(dataset, config=config)
print(attack.attack())

