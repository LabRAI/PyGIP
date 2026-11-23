"""
MDP Defense Example

Demonstrates the Matrix Decomposition + Differential Privacy defense
for protecting GNN models against privacy attacks.

Usage:
    python examples/defense/MDP.py
"""

from pygip.datasets import Cora, CiteSeer, PubMed
from pygip.models.defense import MDP


def main():
    # Load dataset with PyG format (required for MDP)
    print("Loading Cora dataset...")
    dataset = Cora(api_type='pyg')

    print(f"Dataset: {dataset.dataset_name}")
    print(f"Nodes: {dataset.num_nodes}")
    print(f"Features: {dataset.num_features}")
    print(f"Classes: {dataset.num_classes}")

    # Initialize MDP defense
    print("\nInitializing MDP defense...")
    defense = MDP(
        dataset=dataset,
        attack_node_fraction=0.1,
        # MDP parameters
        nc=4,               # Number of federated calculators
        es=2,               # Number of eigenvalue shares
        epsilon=30.0,       # DP privacy budget (higher = less noise)
        keep_ratio=0.8,     # Training data fraction per calculator
        # Model architecture
        hidden_dim=16,
        dropout=0.5,
        # Training
        lr=0.01,
        weight_decay=5e-4,
        epochs=200,
        patience=50,
        seed=42,
    )

    # Execute defense
    print("\nExecuting MDP defense...")
    results = defense.defend()

    # Print results
    print("\n" + "=" * 50)
    print("DEFENSE RESULTS")
    print("=" * 50)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Access internal state if needed
    print("\nDefense Details:")
    print(f"  Adjacency shares: {len(defense.get_adjacency_shares())}")
    stats = defense.get_training_stats()
    if stats:
        print(f"  Training epochs: {stats['epochs_trained']}")
        print(f"  Final val accuracy: {stats['val_acc'][-1]:.4f}")


def run_epsilon_sweep():
    """Sweep over different privacy budgets."""
    print("\n" + "=" * 50)
    print("EPSILON SWEEP")
    print("=" * 50)

    dataset = Cora(api_type='pyg')

    epsilons = [10.0, 20.0, 30.0, float('inf')]

    for eps in epsilons:
        defense = MDP(
            dataset=dataset,
            nc=4,
            es=2,
            epsilon=eps,
            epochs=100,
            patience=30,
            seed=42,
        )

        results = defense.defend()
        acc = results.get('test_acc', 0)

        eps_str = "inf" if eps == float('inf') else f"{eps:.1f}"
        print(f"epsilon={eps_str:>6}: test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
    run_epsilon_sweep()
