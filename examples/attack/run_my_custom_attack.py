from pygip.datasets import Cora
from pygip.models.attack.my_custom_attack import MyCustomAttack
import dgl

dataset = Cora(api_type="dgl", path="./data")
attack = MyCustomAttack(dataset, attack_node_fraction=0.25, samples_per_class=5, subgraph_size=8, seed=42)
result = attack.attack()

print("STATUS:", result.get("status"))
print("Original nodes:", result.get("num_original_nodes"))
print("New nodes added:", result.get("num_new_nodes"))
print("Requested fraction:", result.get("attack_node_fraction_requested"))
print("Actual fraction used:", result.get("attack_node_fraction_used"))
print("Metrics:", result.get("metrics"))
print("Time (s):", result.get("time_seconds"))

perturbed = result.get("perturbed_graph")
if perturbed is not None:
    dgl.save_graphs("perturbed_graph.bin", [perturbed])
    print("Perturbed graph saved to perturbed_graph.bin")

