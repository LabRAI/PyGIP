import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pygip.datasets import Cora, CiteSeer, PubMed
from pygip.models.defense.BackdoorWM import BackdoorWM
from pygip.models.nn import GCN

def train_local_model(model, graph, features, labels, mask, device, epochs=300, lr=0.01):
    """A helper function to train a model locally."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.to(device), features.to(device))
        loss = F.cross_entropy(logits[mask], labels[mask])
        loss.backward()
        optimizer.step()
    return model

def visualize_and_save_results(all_results):
    """
    Creates a single, comprehensive plot with three sub-charts for all metrics
    and saves it as one PNG file.
    """
    dataset_names = [res['name'] for res in all_results]
    n_datasets = len(dataset_names)

    # --- Data Extraction ---
    # Clean Accuracies
    original_accs = [res['results']['original_metrics']['accuracy'] * 100 for res in all_results]
    tcas = [res['results']['TCA'] * 100 for res in all_results]
    eca1s = [res['results']['s1_metrics']['accuracy'] * 100 for res in all_results]
    eca2s = [res['results']['s2_metrics']['accuracy'] * 100 for res in all_results]
    
    # Backdoor Accuracies
    tbas = [res['results']['TBA'] * 100 for res in all_results]
    eba1s = [res['results']['EBA1'] * 100 for res in all_results]
    eba2s = [res['results']['EBA2'] * 100 for res in all_results]

    # F1-Scores
    original_f1s = [res['results']['original_metrics']['f1'] for res in all_results]
    defense_f1s = [res['results']['defense_metrics']['f1'] for res in all_results]
    s1_f1s = [res['results']['s1_metrics']['f1'] for res in all_results]
    s2_f1s = [res['results']['s2_metrics']['f1'] for res in all_results]

    # This is for plotting
    x = np.arange(n_datasets)
    width = 0.2
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 8))
    fig.suptitle('Comprehensive Backdoor Watermark Analysis', fontsize=20)

    # Subplot 1: Clean Accuracy
    ax1.bar(x - 1.5*width, original_accs, width, label='Baseline (Clean)')
    ax1.bar(x - 0.5*width, tcas, width, label='Watermarked (TCA)')
    ax1.bar(x + 0.5*width, eca1s, width, label='Surrogate w/ Coop (ECA1)')
    ax1.bar(x + 1.5*width, eca2s, width, label='Surrogate w/o Coop (ECA2)')
    ax1.set_title('Clean Accuracy Comparison', fontsize=14)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x, dataset_names)
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Subplot 2: Backdoor Accuracy
    ax2.bar(x - width, tbas, width, label='Watermarked (TBA)')
    ax2.bar(x, eba1s, width, label='Surrogate w/ Coop (EBA1)')
    ax2.bar(x + width, eba2s, width, label='Surrogate w/o Coop (EBA2)')
    ax2.set_title('Backdoor Accuracy (Watermark Transfer)', fontsize=14)
    ax2.set_ylabel('Backdoor Accuracy (%)')
    ax2.set_xticks(x, dataset_names)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Subplot 3: F1-Score
    ax3.bar(x - 1.5*width, original_f1s, width, label='Baseline (Clean)')
    ax3.bar(x - 0.5*width, defense_f1s, width, label='Watermarked')
    ax3.bar(x + 0.5*width, s1_f1s, width, label='Surrogate w/ Coop')
    ax3.bar(x + 1.5*width, s2_f1s, width, label='Surrogate w/o Coop')
    ax3.set_title('F1-Score Comparison (on Clean Data)', fontsize=14)
    ax3.set_ylabel('F1-Score (Macro)')
    ax3.set_xticks(x, dataset_names)
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = 'full_defense_analysis_summary.png'
    plt.savefig(filename)
    plt.close()
    print(f"\nComprehensive results visualization saved to '{filename}'")


def main():
    parser = argparse.ArgumentParser(description="Run full analysis of Backdoor Watermarking Defense.")
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer', 'PubMed'], help='List of datasets to run on.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., "cpu", "cuda:0").')
    args = parser.parse_args()

    dataset_map = {'Cora': Cora, 'CiteSeer': CiteSeer, 'PubMed': PubMed}
    verification_threshold = 0.50
    
    all_results = []

    for dataset_name in args.datasets:
        if dataset_name not in dataset_map:
            print(f"Dataset '{dataset_name}' not found. Skipping.")
            continue

        print("\n" + "#"*70)
        print(f"  ANALYZING DATASET: {dataset_name.upper()} on device {args.device.upper()}")
        print("#"*70)

        dataset = dataset_map[dataset_name](api_type='dgl')
        med = BackdoorWM(dataset, attack_node_fraction=0.1, trigger_rate=0.2, l=40, target_label=0)
        med.device = torch.device(args.device)
        med.graph_data = med.graph_data.to(med.device)

        print("\n--- Training Clean Baseline Model ---")
        clean_model = GCN(med.num_features, med.num_classes).to(med.device)
        train_local_model(clean_model, med.graph_data, med.features, med.labels, med.train_mask, med.device)
        original_metrics = med.evaluate_model(clean_model, med.features, med.labels)
        
        print("\n--- Training Watermarked Target Model (F_wm) ---")
        watermarked_model = med.train_target_model()
        defense_metrics = med.evaluate_model(watermarked_model, med.features, med.labels) 
        TCA = defense_metrics['accuracy']
        TBA = med.verify_backdoor(watermarked_model, med.trigger_nodes, med.target_label) 

        print("\n--- Simulating Model Extraction Attacks ---")
        surrogate_model_1 = GCN(med.num_features, med.num_classes).to(med.device)
        with torch.no_grad():
            surrogate_labels_1 = watermarked_model(med.graph_data, med.poisoned_features).argmax(dim=1)
        train_local_model(surrogate_model_1, med.graph_data, med.poisoned_features, surrogate_labels_1, med.train_mask, med.device)
        s1_metrics = med.evaluate_model(surrogate_model_1, med.features, med.labels)
        EBA1 = med.verify_backdoor(surrogate_model_1, med.trigger_nodes, med.target_label)
        
        surrogate_model_2 = GCN(med.num_features, med.num_classes).to(med.device)
        with torch.no_grad():
            surrogate_labels_2 = watermarked_model(med.graph_data, med.features).argmax(dim=1)
        train_local_model(surrogate_model_2, med.graph_data, med.features, surrogate_labels_2, med.train_mask, med.device)
        s2_metrics = med.evaluate_model(surrogate_model_2, med.features, med.labels)
        EBA2 = med.verify_backdoor(surrogate_model_2, med.trigger_nodes, med.target_label)

        
        results_for_viz = {
            'original_metrics': original_metrics, 'TCA': TCA, 'TBA': TBA,
            'defense_metrics': defense_metrics, 's1_metrics': s1_metrics,
            's2_metrics': s2_metrics, 'EBA1': EBA1, 'EBA2': EBA2
        }
        all_results.append({'name': dataset_name, 'results': results_for_viz})

        
        print("\n======================== Final Results =========================================")
        
        print(f"  Baseline Model (Clean):")
        print(f"    - Original Accuracy: {original_metrics['accuracy'] * 100:.2f}%, F1: {original_metrics['f1']:.4f}")
        print(f"\n  Watermarked Model (F_wm):")
        print(f"    - Clean Accuracy (TCA): {TCA * 100:.2f}%")
        print(f"    - Backdoor Accuracy (TBA): {TBA * 100:.2f}%")
        print(f"    - F1 Score (on clean data): {defense_metrics['f1']:.4f}")
        print("\n  Scenario 1: Surrogate Model (With Cooperation)")
        v1_result = "Extracted" if EBA1 > verification_threshold else "Independent"
        print(f"    - Clean Accuracy (ECA): {s1_metrics['accuracy'] * 100:.2f}%")
        print(f"    - Backdoor Accuracy (EBA): {EBA1 * 100:.2f}%")
        print(f"    - Verification Result: {v1_result}")
        print("\n  Scenario 2: Surrogate Model (Without Cooperation)")
        v2_result = "Extracted" if EBA2 > verification_threshold else "Independent"
        print(f"    - Clean Accuracy (ECA): {s2_metrics['accuracy'] * 100:.2f}%")
        print(f"    - Backdoor Accuracy (EBA): {EBA2 * 100:.2f}%")
        print(f"    - Verification Result: {v2_result}")
        print("==============================================================================")
    
    # After the loop, create the single summary image So that the user can know it in a glance
    if all_results:
        visualize_and_save_results(all_results)

if __name__ == '__main__':
    main()