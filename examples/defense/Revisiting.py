from pygip.datasets import Cora
from pygip.models.defense import Revisiting


def main():
    # Load dataset
    dataset = Cora()

    # Init defense (tweak params as needed)
    defense = Revisiting(
        dataset,
        attack_node_fraction=0.20,  # fraction of nodes to mix
        alpha=0.80,  # neighbor-mixing strength [0,1]
    )

    print("Initialized Revisiting defense; starting defend()...")
    results = defense.defend()
    print("Defense finished. Results:", results)


if __name__ == "__main__":
    main()
