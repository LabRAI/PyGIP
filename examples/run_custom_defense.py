# examples/run_custom_defense.py
import argparse
from pygip.datasets.datasets import Dataset
from src.custom_defense import NeighborSmoothingDefense

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dataset = Dataset(api_type="pyg", path="./data")
    defense = NeighborSmoothingDefense(dataset, attack_node_fraction=args.fraction, device=args.device)
    results = defense.defend(retrain_epochs=50, seed=0)
    print("Defense results:", results)

if __name__ == "__main__":
    main()
