
import torch
from gnnfingers_pygip.datasets.aids import AIDSMatchingDataset
from gnnfingers_pygip.attacks.gnnfingers_aids import GNNFingersAttack
from gnnfingers_pygip.defenses.gnnfingers_aids_defense import GNNFingersDefense
from gnnfingers_pygip.eval import evaluate
from gnnfingers_pygip.utils import get_device

def run_gnnfingers():
    ds = AIDSMatchingDataset(api_type='pyg', path='./data',
                             num_train_pairs=1400, num_val_pairs=200, num_test_pairs=200)
    attack = GNNFingersAttack(ds, attack_node_fraction=0.1, device=get_device(),
                              victim_hidden=64, victim_out=64)
    atk_res = attack.attack()
    defense = GNNFingersDefense(ds, attack_node_fraction=0.1, device=get_device(),
                                P=64, fp_nodes=32, backbone_out=64, joint_iters=150,
                                verif_lr=5e-4, fp_lr=2e-3)
    def_res = defense.defend(attack_results=atk_res)
    metrics = evaluate(defense, atk_res['positive_models'], atk_res['negative_models'])
    print(f"TPR: {metrics['robustness_TPR']:.4f}, TNR: {metrics['uniqueness_TNR']:.4f}, ARUC: {metrics['ARUC']:.4f}, Mean Test Accuracy: {metrics['mean_test_accuracy']:.4f}")

if __name__ == '__main__':
    run_gnnfingers()
