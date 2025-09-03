
# Example: GNNFingers on ENZYMES (Graph Classification) - CPU friendly
import torch
from datasets.enzymes import EnzymesDataset
from attacks.gnnfingers_enzymes_gc import GNNFingersAttack
from defenses.gnnfingers_enzymes_gc import GNNFingersDefense, evaluate

def run():
    ds = EnzymesDataset(api_type='pyg', path='./data')
    attack = GNNFingersAttack(ds, attack_node_fraction=0.1, device='cuda' if torch.cuda.is_available() else 'cpu',
                              victim_hidden=128, victim_out=128)
    atk = attack.attack()
    defense = GNNFingersDefense(ds, attack_node_fraction=0.1, device='cuda' if torch.cuda.is_available() else 'cpu',
                                P=64, fp_nodes=32, backbone_out=64, joint_iters=150,
                                verif_lr=5e-4, fp_lr=2e-3)
    _ = defense.defend(atk)
    print("\n📈 Evaluating across all suspects...")
    metrics = evaluate(defense, atk['positive_models'], atk['negative_models'])
    print(f"TPR: {metrics['robustness_TPR']:.4f}, TNR: {metrics['uniqueness_TNR']:.4f}, ARUC: {metrics['ARUC']:.4f}, Mean Test Acc: {metrics['mean_test_accuracy']:.4f}")

if __name__ == '__main__':
    run()
