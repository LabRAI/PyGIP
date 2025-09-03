import torch
from utils.common import get_device, set_seed
from datasets.citeseer import CiteseerDataset
from attacks.gnnfingers_citeseer_nc import GNNFingersAttack
from defenses.gnnfingers_citeseer_nc import GNNFingersDefense, evaluate

def main():
    set_seed(7)
    print("🚀 PyGIP-compliant GNNFingers on CiteSeer (Node Classification)")
    ds = CiteseerDataset(api_type='pyg', path='./data')
    print(f"Dataset: {ds.get_name()} | nodes={ds.num_nodes} | feat={ds.num_features} | classes={ds.num_classes}")
    attack = GNNFingersAttack(ds, attack_node_fraction=0.1, device=get_device(),
                              victim_hidden=64, victim_out=64)
    atk = attack.attack()
    defense = GNNFingersDefense(ds, attack_node_fraction=0.1, device=get_device(),
                                fp_nodes=100, backbone_out=64, joint_iters=150,
                                verif_lr=5e-4, fp_lr=2e-3)
    defense_res = defense.defend(attack_results=atk)
    print("\n🔎 Sanity check")
    print("Positive:", defense.verify_ownership(atk['positive_models'][0]))
    print("Negative:", defense.verify_ownership(atk['negative_models'][0]))
    print("\n📈 Eval...")
    metrics = evaluate(defense, atk['positive_models'], atk['negative_models'])
    print({k: round(v, 4) for k, v in metrics.items()})

if __name__ == "__main__":
    main()
