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


# this all those print statements are used inoreder to provide a visual summary of the results
def generate_summary_plot(all_results, max_drop_threshold):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for dataset_name, results in all_results.items():
        baseline_acc = results[0.0]
        ratios = sorted(results.keys())
        accuracies = [results[r] for r in ratios]

        break_index = len(accuracies)
        for i, acc in enumerate(accuracies):
            if (baseline_acc - acc) > max_drop_threshold:
                break_index = i + 1
                break
        
        ratios_to_plot = ratios[:break_index]
        accuracies_to_plot = [acc * 100 for acc in accuracies[:break_index]]
        ratios_percent = [r * 100 for r in ratios_to_plot]

        line, = ax.plot(ratios_percent, accuracies_to_plot, marker='o', linestyle='-', label=f'{dataset_name} Accuracy')
        ax.axhline(y=baseline_acc * 100, color=line.get_color(), linestyle='--', 
                   label=f'{dataset_name} Baseline ({baseline_acc*100:.2f}%)')

    ax.set_title('Effective Range of Graph Pruning Before Performance Drop', fontsize=16)
    ax.set_xlabel('Pruning Ratio (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_path = "pruning_effective_range_summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Focused summary plot saved to: {save_path}")


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
        
        baseline_defense = GraphPruningDefense(dataset, 
                                               pruning_ratio=0.0,
                                               attack_node_fraction=CONFIG["attack_node_fraction"])
        baseline_results = baseline_defense.defend()
        baseline_accuracy = baseline_results['defended_accuracy']
        print(f"     -> Baseline Accuracy for {dataset_name}: {baseline_accuracy * 100:.2f}%")
        
        print(f"\n   - Sub-step [2/3]: Searching for optimal pruning ratio...")
        current_ratio = CONFIG["initial_pruning_ratio"]
        run_results = {0.0: baseline_accuracy}
        
        while True:
            defense = GraphPruningDefense(dataset, 
                                          pruning_ratio=current_ratio,
                                          attack_node_fraction=CONFIG["attack_node_fraction"])
            results_dict = defense.defend()
            current_accuracy = results_dict['defended_accuracy']
            run_results[current_ratio] = current_accuracy
            accuracy_drop = baseline_accuracy - current_accuracy

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
        optimal_accuracy = baseline_accuracy
        for ratio, acc in run_results.items():
            if (baseline_accuracy - acc) <= CONFIG["max_accuracy_drop"] and ratio > optimal_ratio:
                optimal_ratio = ratio
                optimal_accuracy = acc
        
        print("     " + "="*60)
        print(f"      Baseline Accuracy:         {baseline_accuracy * 100:.2f}%")
        print(f"      Acceptable Accuracy Drop:  < {CONFIG['max_accuracy_drop'] * 100:.2f}%")
        print("     " + "-" * 60)
        print(f"      Optimal Pruning Ratio:     {optimal_ratio * 100:.1f}%")
        print(f"      Accuracy at this Ratio:    {optimal_accuracy * 100:.2f}%")
        print(f"      Final Accuracy Drop:       {(baseline_accuracy - optimal_accuracy) * 100:.2f}%")
        print("     " + "="*60)

    step_num = num_total_steps
    print("\n" + "="*80)
    print(f"      STEP [{step_num}/{num_total_steps}]: Generating Overall Summary Plot")
    print("="*80)
    generate_summary_plot(all_results_for_plot, CONFIG["max_accuracy_drop"])
    print("="*80)

if __name__ == "__main__":
    main()