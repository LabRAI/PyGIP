from pygip.datasets import Cora
from pygip.models.defense.my_custom_defense import MyCustomDefense
import dgl
import time

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = Cora(api_type="dgl", path="./data")

    print("Initializing MyCustomDefense...")
    defense = MyCustomDefense(
        dataset=dataset,
        attack_node_fraction=0.25,
        hidden_dim=64,
        epochs=100,
        lr=0.01,
        weight_decay=5e-4
    )

    print("Starting defense procedure...")
    start = time.time()
    result = defense.defend()
    elapsed = time.time() - start

    print("=" * 60)
    print("Defense Report")
    print("=" * 60)
    print(f"Status: {result.get('status')}")
    print(f"Target model trained: {result.get('target_model_trained')}")
    print(f"Defense model trained: {result.get('defense_model_trained')}")
    print(f"Time (s): {elapsed:.3f}")
    print("-" * 60)

    metrics = result.get("metrics", {})
    print(f"TCA (Target Clean Accuracy): {metrics.get('TCA')}")
    print(f"ECA (Extracted Clean Accuracy): {metrics.get('ECA')}")
    print(f"TBA (Target Backdoored Accuracy): {metrics.get('TBA')}")
    print(f"EBA (Extracted Backdoored Accuracy): {metrics.get('EBA')}")
    print(f"Fidelity (Agreement): {metrics.get('Fidelity')}")
    print("-" * 60)

    attack_target = result.get("attack_result_on_target", {})
    perturbed = attack_target.get("perturbed_graph")
    if perturbed is not None:
        dgl.save_graphs("defended_perturbed_graph.bin", [perturbed])
        print("Perturbed graph saved to defended_perturbed_graph.bin")

    print("=" * 60)
    print("Defense completed successfully.")
