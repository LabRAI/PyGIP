# examples/citeseer_linkprediction.py
# Runnable example for Citeseer (Link Prediction) with GNNFingers

import torch
from utils.common import get_device
from datasets.citeseer import CiteseerDataset
from attacks.gnnnfingers_citeseer_lp import GNNFingersAttack
from defenses.gnnnfingers_citeseer_lp import GNNFingersDefense

def main():
    print("🚀 Starting PyGIP-compliant GNNFingers on Citeseer (Link Prediction)")
    device = get_device()
    # Dataset
    ds = CiteseerDataset(api_type='pyg', path='./data')
    print(f"Dataset: {ds.get_name()} | nodes={ds.num_nodes} | feat={ds.num_features}")
    # Attack
    print("\n🔨 Starting attack phase...")
    attack = GNNFingersAttack(ds, attack_node_fraction=0.1, device=device)
    atk = attack.attack()
    # Defense
    print("\n🛡️ Starting defense phase...")
    defense = GNNFingersDefense(ds, attack_node_fraction=0.1, device=device)
    out = defense.defend(atk)
    # Evaluation
    print("\n📈 Evaluation results:")
    metrics = defense.evaluate(out['pos_test'], out['neg_test'])
    print(metrics)

if __name__ == '__main__':
    main()
