import matplotlib.pyplot as plt
from pygip import datasets
from pygip.models.defense import GraphPruningDefense

CONFIG = {
    "datasets_to_run": ["Cora", "CiteSeer", "PubMed"],
    "initial_pruning_ratio": 0.05,
    "pruning_step": 0.02,
    "max_accuracy_drop": 0.02,
    "attack_node_fraction": 0.2
}

def generate_summary_plot(all_results, max_drop_threshold):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(len(all_results), 2, figsize=(14, 4 * len(all_results)))
    if len(all_results) == 1:
        axs = [axs]

    for ax_row, (dataset_name, results) in zip(axs, all_results.items()):
        ratios = sorted(results.keys())
        ecas = [results[r]['ECA'] * 100 for r in ratios]
        fidelities = [results[r]['Fidelity'] * 100 for r in ratios]
        ratios_percent = [r * 100 for r in ratios]

        ax_e = ax_row[0]
        ax_e.plot(ratios_percent, ecas, marker='o', linestyle='-')
        baseline = results.get(0.0, {}).get('baseline_accuracy', None)
        if baseline is not None:
            ax_e.axhline(y=baseline * 100, linestyle='--', label=f'Baseline ({baseline*100:.2f}%)')
        ax_e.set_title(f'{dataset_name} - Test Accuracy (ECA) vs Pruning Ratio')
        ax_e.set_xlabel('Pruning Ratio (%)'); ax_e.set_ylabel('Test Accuracy (%)')
        ax_e.grid(True); ax_e.legend()

        ax_f = ax_row[1]
        ax_f.plot(ratios_percent, fidelities, marker='s', linestyle='-')
        ax_f.set_title(f'{dataset_name} - Fidelity vs Pruning Ratio')
        ax_f.set_xlabel('Pruning Ratio (%)'); ax_f.set_ylabel('Fidelity (%)')
        ax_f.grid(True)

    plt.tight_layout()
    save_path = "pruning_ECA_fidelity_summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nSummary plot saved to: {save_path}")

def main():
    all_results_for_plot = {}
    datasets_to_run = CONFIG["datasets_to_run"]
    num_total_steps = len(datasets_to_run) + 1

    for i, dataset_name in enumerate(datasets_to_run):
        step_num = i + 1
        print("\n" + "="*80)
        print(f"      STEP [{step_num}/{num_total_steps}]: Running Experiment for {dataset_name.upper()}")
        print("="*80)

        print(f"\n   - Sub-step [1/3]: Loading {dataset_name} and calculating baseline...")
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class()
        
        baseline_defense = GraphPruningDefense(dataset,attack_node_fraction=["attack_node_fraction"],pruning_ratio=0.0)
        baseline_results = baseline_defense.defend()
        baseline_accuracy = baseline_results['baseline_accuracy']
        print(f"     -> Baseline Accuracy for {dataset_name}: {baseline_accuracy * 100:.2f}%")
        
        print(f"\n   - Sub-step [2/3]: Searching for optimal pruning ratio...")
        current_ratio = CONFIG["initial_pruning_ratio"]
        run_results = {0.0: baseline_results}
        
        while True:
            defense = GraphPruningDefense(dataset,attack_node_fraction=CONFIG["attack_node_fraction"], pruning_ratio=current_ratio)
            results_dict = defense.defend()
            run_results[current_ratio] = results_dict
            current_accuracy = results_dict['ECA']
            accuracy_drop = baseline_accuracy - current_accuracy

            print(f"Using device: {defense.device}")
            print(f"   -> Metrics for {dataset_name} (ratio={current_ratio*100:.1f}%):")
            print(f"      ECA: {results_dict['ECA']*100:.2f}% | Fidelity: {results_dict['Fidelity']*100:.2f}% | Edges removed: {results_dict['edges_removed']}")
            
            if accuracy_drop <= CONFIG["max_accuracy_drop"]:
                print(f"      -> Success! Drop: {accuracy_drop*100:.2f}%. Continuing search.")
                current_ratio = round(current_ratio + CONFIG["pruning_step"], 2)
            else:
                print(f"      -> Stopping. Drop of {accuracy_drop*100:.2f}% exceeded threshold.")
                break
            if current_ratio >= 0.8:
                print("      -> Stopping. Reached maximum pruning limit.")
                break
        
        all_results_for_plot[dataset_name] = run_results

        print(f"\n   - Sub-step [3/3]: Optimal Pruning Results for {dataset_name}")
        optimal_ratio = 0.0
        optimal_metrics = baseline_results
        for ratio, metrics in run_results.items():
            if (baseline_accuracy - metrics['ECA']) <= CONFIG["max_accuracy_drop"] and ratio > optimal_ratio:
                optimal_ratio = ratio
                optimal_metrics = metrics
        
        print("     " + "="*60)
        print(f"      Baseline Accuracy:         {baseline_accuracy * 100:.2f}%")
        print(f"      Acceptable Accuracy Drop:  < {CONFIG['max_accuracy_drop'] * 100:.2f}%")
        print("     " + "-" * 60)
        print(f"      Optimal Pruning Ratio:     {optimal_ratio * 100:.1f}%")
        print(f"      Accuracy at this Ratio:    {optimal_metrics['ECA'] * 100:.2f}%")
        print(f"      Fidelity at this Ratio:    {optimal_metrics['Fidelity'] * 100:.2f}%")
        print(f"      Edges removed:             {optimal_metrics['edges_removed']}")
        print("     " + "="*60)

    print("\n" + "="*80)
    print(f"      STEP [{num_total_steps}/{num_total_steps}]: Generating Overall Summary Plot")
    print("="*80)
    generate_summary_plot(all_results_for_plot, CONFIG["max_accuracy_drop"])
    print("="*80)

if __name__ == "__main__":
    main()
