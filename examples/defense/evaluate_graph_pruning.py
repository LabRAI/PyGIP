import matplotlib.pyplot as plt
from pygip import datasets
from pygip.models.defense import GraphPruningDefense


# This dictionary holds all the parameters for the experiment.
CONFIG = {
    "datasets_to_run": ["Cora", "CiteSeer", "PubMed"],
    "initial_pruning_ratio": 0.05,
    "pruning_step": 0.02,
    "max_accuracy_drop": 0.02,
    "attack_node_fraction": 0.2
}

def main():
    all_results_for_plot = {}
    datasets_to_run = CONFIG["datasets_to_run"]
    
    # Process each dataset specified in the CONFIG.
    for i, dataset_name in enumerate(datasets_to_run):
        print(f"Processing dataset: {dataset_name}")
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class()
        
        defense = GraphPruningDefense(dataset, attack_node_fraction=CONFIG["attack_node_fraction"], pruning_ratio=0.0)
        
        baseline_results = defense.defend()
        baseline_accuracy = baseline_results['baseline_accuracy']
        
        current_ratio = CONFIG["initial_pruning_ratio"]
        run_results = {0.0: baseline_results}


        # This loop iteratively increases the pruning ratio and stops when accuracy drops too much.        
        while True:
            defense.pruning_ratio = current_ratio
            
            results_dict = defense.defend()
            run_results[current_ratio] = results_dict
            current_accuracy = results_dict['ECA']
            accuracy_drop = baseline_accuracy - current_accuracy

            if accuracy_drop <= CONFIG["max_accuracy_drop"]:
                current_ratio = round(current_ratio + CONFIG["pruning_step"], 2)
            else:
                break
            if current_ratio >= 0.8:
                break
        
        all_results_for_plot[dataset_name] = run_results

        optimal_ratio = 0.0
        optimal_metrics = baseline_results
        for ratio, metrics in run_results.items():
            if (baseline_accuracy - metrics['ECA']) <= CONFIG["max_accuracy_drop"] and ratio > optimal_ratio:
                optimal_ratio = ratio
                optimal_metrics = metrics

        print(f"    -> Optimal Pruning Ratio: {optimal_ratio*100:.1f}%")
        print(f"    -> Metrics for {dataset_name} (ratio={optimal_ratio*100:.1f}%):")
        print(f"        ECA: {optimal_metrics['ECA']*100:.2f}% | Fidelity: {optimal_metrics['Fidelity']*100:.2f}% | Edges removed: {optimal_metrics['edges_removed']}")
        print("="*80)


if __name__ == "__main__":
    main()