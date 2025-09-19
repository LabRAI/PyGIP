# examples/run_custom_attack.py
import argparse
from pygip.datasets.datasets import Dataset
from pygip.src.custom_attack import FeatureFlipAttack

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dataset = Dataset(api_type="pyg", path="./data")
    attack = FeatureFlipAttack(dataset, attack_node_fraction=args.fraction, model_path=None, device=args.device)
    results = attack.attack(retrain_target=True, retrain_epochs=50, seed=0)
    print("Attack results:", results)

if __name__ == "__main__":
    main()
