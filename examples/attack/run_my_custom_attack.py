from pygip.datasets import Cora
from pygip.models.attack.my_custom_attack import MyCustomAttack
import dgl

# Load dataset
dataset = Cora(api_type="dgl", path="./data")

# Initialize custom attack
attack = MyCustomAttack(
    dataset,
    attack_node_fraction=0.25,   # 25% of nodes allowed for attack
    samples_per_class=5,         # synthetic subgraphs per class
    subgraph_size=8,             # nodes per synthetic subgraph
    seed=42
)

# Run attack
result = attack.attack()

print("="*60)
print("Attack Report")
print("="*60)
print("Status:", result.get("status"))
print("Original nodes:", result.get("num_original_nodes"))
print("New nodes added:", result.get("num_new_nodes"))
print("Requested fraction:", result.get("attack_node_fraction_requested"))
print("Actual fraction used:", result.get("attack_node_fraction_used"))
print("Time (s):", result.get("time_seconds"))
print("-"*60)

metrics = result.get("metrics", {})
print("TCA (Target Clean Accuracy):", metrics.get("TCA"))
print("ECA (Extracted Clean Accuracy):", metrics.get("ECA"))
print("TBA (Target Backdoored Accuracy):", metrics.get("TBA"))
print("EBA (Extracted Backdoored Accuracy):", metrics.get("EBA"))
print("Fidelity (Surrogate vs Target):", metrics.get("fidelity"))
print("-"*60)

# Save perturbed graph for later use
perturbed = result.get("perturbed_graph")
if perturbed is not None:
    dgl.save_graphs("perturbed_graph.bin", [perturbed])
    print("Perturbed graph saved to perturbed_graph.bin")
