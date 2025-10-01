import argparse
import matplotlib.pyplot as plt
import numpy as np

from pygip.datasets.datasets import Cora, CiteSeer, PubMed 
from pygip.models.defense.backdoor_wm_defense1 import BackdoorWatermarkDefense

def visualize_combined_results(all_results):
    """
    Creates a single bar chart with subplots for all dataset results 
    and saves it as one PNG file.
    """
    dataset_names = [res['name'] for res in all_results]
    n_datasets = len(dataset_names)
    
    # Extract all metrics for clean and backdoor accuracy
    clean_accuracies = {
        'Target': [res['results']['target_model_metrics']['TCA'] * 100 for res in all_results],
        'Surrogate (Coop)': [res['results']['scenario_with_cooperation']['ECA'] * 100 for res in all_results],
        'Surrogate (No Coop)': [res['results']['scenario_without_cooperation']['ECA'] * 100 for res in all_results]
    }
    
    backdoor_accuracies = {
        'Target': [res['results']['target_model_metrics']['TBA'] * 100 for res in all_results],
        'Surrogate (Coop)': [res['results']['scenario_with_cooperation']['EBA'] * 100 for res in all_results],
        'Surrogate (No Coop)': [res['results']['scenario_without_cooperation']['EBA'] * 100 for res in all_results]
    }

    # --- NEW: Extract Fidelity metrics ---
    fidelities = {
        'Surrogate (Coop)': [res['results']['scenario_with_cooperation']['Fidelity'] * 100 for res in all_results],
        'Surrogate (No Coop)': [res['results']['scenario_without_cooperation']['Fidelity'] * 100 for res in all_results]
    }

    x = np.arange(n_datasets)  # the label locations
    
    # --- CHANGED: Create 3 subplots instead of 2 ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Backdoor Watermarking Defense Summary', fontsize=16)

    # --- Subplot 1: Clean Accuracy (No changes here) ---
    width1 = 0.25
    rects1 = ax1.bar(x - width1, clean_accuracies['Target'], width1, label='Target Model')
    rects2 = ax1.bar(x, clean_accuracies['Surrogate (Coop)'], width1, label='Surrogate (With Coop.)')
    rects3 = ax1.bar(x + width1, clean_accuracies['Surrogate (No Coop)'], width1, label='Surrogate (Without Coop.)')
    ax1.set_title('Clean Accuracy (TCA / ECA)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x, dataset_names)
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.bar_label(rects1, padding=3, fmt='%.1f')
    ax1.bar_label(rects2, padding=3, fmt='%.1f')
    ax1.bar_label(rects3, padding=3, fmt='%.1f')

    # --- Subplot 2: Backdoor Accuracy (No changes here) ---
    rects4 = ax2.bar(x - width1, backdoor_accuracies['Target'], width1, label='Target Model')
    rects5 = ax2.bar(x, backdoor_accuracies['Surrogate (Coop)'], width1, label='Surrogate (With Coop.)')
    rects6 = ax2.bar(x + width1, backdoor_accuracies['Surrogate (No Coop)'], width1, label='Surrogate (Without Coop.)')
    ax2.set_title('Backdoor Accuracy (TBA / EBA)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticks(x, dataset_names)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.bar_label(rects4, padding=3, fmt='%.1f')
    ax2.bar_label(rects5, padding=3, fmt='%.1f')
    ax2.bar_label(rects6, padding=3, fmt='%.1f')

    # --- NEW: Subplot 3: Fidelity ---
    width2 = 0.35
    rects7 = ax3.bar(x - width2/2, fidelities['Surrogate (Coop)'], width2, label='Surrogate (With Coop.)')
    rects8 = ax3.bar(x + width2/2, fidelities['Surrogate (No Coop)'], width2, label='Surrogate (Without Coop.)')
    ax3.set_title('Fidelity')
    ax3.set_ylabel('Agreement with Target Model (%)')
    ax3.set_xticks(x, dataset_names)
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.bar_label(rects7, padding=3, fmt='%.1f')
    ax3.bar_label(rects8, padding=3, fmt='%.1f')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = 'backdoor_defense_summary.png'
    plt.savefig(filename)
    plt.close()
    print(f"\nCombined results visualization saved to '{filename}'")


def main():
    # This function remains the same as the previous version
    parser = argparse.ArgumentParser(description="Run Backdoor Watermarking Defense.")
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        default=['Cora', 'CiteSeer', 'PubMed'], 
        help='List of datasets to run the defense on.'
    )
    args = parser.parse_args()
    thresholds = {'cora': 0.50, 'citeseer': 0.53, 'pubmed': 0.50}
    dataset_map = {'Cora': Cora, 'CiteSeer': CiteSeer, 'PubMed': PubMed}
    
    all_results = []

    for dataset_name in args.datasets:
        if dataset_name not in dataset_map:
            print(f"Dataset '{dataset_name}' not recognized. Skipping. Available: {list(dataset_map.keys())}")
            continue
            
        print("\n" + "="*50)
        print(f"  Running Backdoor Watermarking Defense on {dataset_name}")
        print("="*50)
        
        dataset = dataset_map[dataset_name](api_type='pyg')
        
        defense = BackdoorWatermarkDefense(dataset=dataset, verification_thresholds=thresholds)
        
        results = defense.defend()
        all_results.append({'name': dataset_name, 'results': results})
        
        print("\n--- Final Results ---")
        print(f"Verification Threshold: {results['verification_threshold']:.2f}\n")
        tca = results['target_model_metrics']['TCA'] * 100
        tba = results['target_model_metrics']['TBA'] * 100
        print(f"Target Model (F_wm):")
        print(f"  - Clean Accuracy (TCA): {tca:.2f}%")
        print(f"  - Backdoor Accuracy (TBA): {tba:.2f}%")
        
        s1 = results['scenario_with_cooperation']
        eca1, eba1, fid1 = s1['ECA'] * 100, s1['EBA'] * 100, s1['Fidelity'] * 100
        print(f"\nSurrogate Model (With Attacker Cooperation):")
        print(f"  - Clean Accuracy (ECA): {eca1:.2f}%")
        print(f"  - Backdoor Accuracy (EBA): {eba1:.2f}%")
        print(f"  - Fidelity: {fid1:.2f}%")
        print(f"  - Verification Result: {s1['verification_result']} (EBA is {'ABOVE' if eba1 > results['verification_threshold'] else 'BELOW'} threshold)")
        
        s2 = results['scenario_without_cooperation']
        eca2, eba2, fid2 = s2['ECA'] * 100, s2['EBA'] * 100, s2['Fidelity'] * 100
        print(f"\nSurrogate Model (Without Attacker Cooperation):")
        print(f"  - Clean Accuracy (ECA): {eca2:.2f}%")
        print(f"  - Backdoor Accuracy (EBA): {eba2:.2f}%")
        print(f"  - Fidelity: {fid2:.2f}%")
        print(f"  - Verification Result: {s2['verification_result']} (EBA is {'ABOVE' if eba2 > results['verification_threshold'] else 'BELOW'} threshold)")

        print("="*50)
        
    # After the defense is run on all datasets, we can visualize the combined results
    if all_results:
        visualize_combined_results(all_results)

if __name__ == "__main__":
    main()
    
    
