
# Example script to run GNNFingers on PROTEINS dataset
import torch
from datasets.proteins import ProteinsDataset
from attacks.gnnfingers_proteins_gc import GNNFingersAttack
from defenses.gnnfingers_proteins_gc import GNNFingersDefense, evaluate

def run_gnnfingers():
    ds = ProteinsDataset(api_type='pyg', path='./data')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attack = GNNFingersAttack(ds, attack_node_fraction=0.1, device=device,
                              victim_hidden=64, victim_out=64)
    atk_res = attack.attack()
    defense = GNNFingersDefense(ds, attack_node_fraction=0.1, device=device,
                                P=64, fp_nodes=32, backbone_out=64, joint_iters=150,
                                verif_lr=5e-4, fp_lr=2e-3)
    _ = defense.defend(attack_results=atk_res)
    metrics = evaluate(defense, atk_res['positive_models'], atk_res['negative_models'])
    print(f"TPR: {metrics['robustness_TPR']:.4f}, TNR: {metrics['uniqueness_TNR']:.4f}, "
          f"ARUC: {metrics['ARUC']:.4f}, Mean Test Acc: {metrics['mean_test_accuracy']:.4f}")

if __name__ == '__main__':
    run_gnnfingers()
