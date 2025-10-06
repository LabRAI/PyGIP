
import torch
from pygip.datasets import Cora
from pygip.models.defense import GNN_Fingers_Multi_Graph_Support

def gnnfingers():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    dataset = Cora(api_type="dgl")
    defense = GNN_Fingers_Multi_Graph_Support.GNNFingers(
        dataset=dataset,
        device=device,
        num_fingerprints=32,
        fingerprint_nodes=64,
        epochs=100,
        task_type='node_level' 
    )
    results = defense.defend()
    print("\n=== Defense Results ===")
    print(f"Target Accuracy: {results.get('target_accuracy', 0):.4f}")
    print(f"Suspect Accuracy: {results.get('suspect_accuracy', 0):.4f}")
    print(f"Verification Result: {results.get('verification_result', 'Unknown')}")


if __name__ == "__main__":
    gnnfingers()