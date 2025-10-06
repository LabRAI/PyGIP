import argparse
import torch
import torch.nn.functional as F
from pygip.datasets.datasets import Cora, CiteSeer, PubMed
from pygip.models.defense.BackdoorWM import BackdoorWM
from pygip.models.nn import GCN

def train_local_model(model, graph, features, labels, mask, device, epochs=300, lr=0.01):
    """A helper function to train a model locally."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.to(device), features.to(device))
        loss = F.cross_entropy(logits[mask], labels[mask])
        loss.backward()
        optimizer.step()
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Backdoor Watermarking Defense.")
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer', 'PubMed'], help='List of datasets to run on.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., "cpu", "cuda:0").')
    args = parser.parse_args()

    dataset_map = {'Cora': Cora, 'CiteSeer': CiteSeer, 'PubMed': PubMed}
    verification_threshold = 0.50

    for dataset_name in args.datasets:
        if dataset_name not in dataset_map:
            continue
        
        print(f"\n--- Results for: {dataset_name} ---")
        dataset = dataset_map[dataset_name](api_type='dgl')
        med = BackdoorWM(dataset, attack_node_fraction=0.1, trigger_rate=0.2, l=40, target_label=0)
        med.device = torch.device(args.device)
        med.graph_data = med.graph_data.to(med.device)

        clean_model = GCN(med.num_features, med.num_classes).to(med.device)

        # --- Watermarked Model Training & Verification ---
        # Train the target model with the backdoor watermark embedded.
        train_local_model(clean_model, med.graph_data, med.features, med.labels, med.train_mask, med.device)
        
        watermarked_model = med.train_target_model()
        TBA = med.verify_backdoor(watermarked_model, med.trigger_nodes, med.target_label) 

        # --- Surrogate Model 1: Cooperative Attacker ---
        # Simulate a model extraction attack where the attacker knows about the poisoned features.
        surrogate_model_1 = GCN(med.num_features, med.num_classes).to(med.device)
        with torch.no_grad():
            surrogate_labels_1 = watermarked_model(med.graph_data, med.poisoned_features).argmax(dim=1)
        train_local_model(surrogate_model_1, med.graph_data, med.poisoned_features, surrogate_labels_1, med.train_mask, med.device)
        EBA1 = med.verify_backdoor(surrogate_model_1, med.trigger_nodes, med.target_label)
        
        # --- Surrogate Model 2: Non-Cooperative Attacker ---
        # Simulate an attack where the attacker only has black-box access and uses clean data.
        surrogate_model_2 = GCN(med.num_features, med.num_classes).to(med.device)
        with torch.no_grad():
            surrogate_labels_2 = watermarked_model(med.graph_data, med.features).argmax(dim=1)
        train_local_model(surrogate_model_2, med.graph_data, med.features, surrogate_labels_2, med.train_mask, med.device)
        EBA2 = med.verify_backdoor(surrogate_model_2, med.trigger_nodes, med.target_label)

        # If the extracted backdoor accuracy is above the threshold, the watermark is considered "Extracted". Otherwise, the surrogate model is considered "Independent" of the watermark.
        v1_result = "Extracted" if EBA1 > verification_threshold else "Independent"
        v2_result = "Extracted" if EBA2 > verification_threshold else "Independent"
        
        print(f"  Watermarked Model -> Backdoor Accuracy (TBA): {TBA * 100:.2f}%")
        print(f"  Surrogate w/ Coop -> Backdoor Accuracy (EBA1): {EBA1 * 100:.2f}%  | Verification: {v1_result}")
        print(f"  Surrogate w/o Coop -> Backdoor Accuracy (EBA2): {EBA2 * 100:.2f}% | Verification: {v2_result}")

if __name__ == '__main__':
    main()