# examples/cora_nodeclassification.py
# Runnable example for Cora Node Classification with GNNFingers (PyGIP-style)

import torch
from datasets.cora import CoraDataset
from attacks.gnnnfingers_cora_node import GNNFingersAttack, get_device, AvgMeter
from defenses.gnnnfingers_cora_node import GNNFingersDefense

def evaluate(defense, pos_models, neg_models):
    print("Evaluating...")
    # Robustness (TPR) & Uniqueness (TNR)
    tpr, tnr = 0.0, 0.0
    for m in pos_models:
        r = defense.verify_ownership(m)
        tpr += r['is_stolen']
    for m in neg_models:
        r = defense.verify_ownership(m)
        tnr += (1 - r['is_stolen'])
    tpr /= max(1, len(pos_models))
    tnr /= max(1, len(neg_models))
    aruc = 0.5 * (tpr + tnr)

    # Mean test accuracy of target victim
    target = defense.target_model
    target.eval()
    acc = AvgMeter()
    with torch.no_grad():
        data = defense.graph_data.to(defense.device)
        test_mask = defense.dataset.test_mask.to(defense.device)
        logits = target(data)
        pred = logits[test_mask].argmax(dim=1)
        acc.add((pred == data.y[test_mask]).float().mean().item())

    return {'robustness_TPR': tpr, 'uniqueness_TNR': tnr, 'ARUC': aruc, 'mean_test_accuracy': acc.avg}

def main():
    print("🚀 Starting PyGIP-compliant GNNFingers on Cora (Node Classification)")
    device = get_device()

    # Dataset
    ds = CoraDataset(api_type='pyg', path='./data')
    print(f"Dataset: {ds.get_name()} | nodes={ds.num_nodes} | feat={ds.num_features} | classes={ds.num_classes}")

    # Attack (victim + suspects)
    attack = GNNFingersAttack(ds, attack_node_fraction=0.1, device=device, victim_hidden=256, victim_out=64)
    atk = attack.attack()

    # Defense (joint training)
    defense = GNNFingersDefense(ds, attack_node_fraction=0.1, device=device,
                                P=64, fp_nodes=32, backbone_out=64, joint_iters=150,
                                verif_lr=5e-4, fp_lr=2e-3)
    _ = defense.defend(attack_results=atk)

    # Quick sanity
    ex_pos = defense.verify_ownership(atk['positive_models'][0])
    ex_neg = defense.verify_ownership(atk['negative_models'][0])
    print("\n🔎 Example verification")
    print("Positive:", ex_pos)
    print("Negative:", ex_neg)

    # Full evaluation
    print("\n📈 Evaluating across suspects...")
    metrics = evaluate(defense, atk['positive_models'], atk['negative_models'])
    print("Results:")
    print(f"  robustness_TPR: {metrics['robustness_TPR']:.4f}")
    print(f"  uniqueness_TNR: {metrics['uniqueness_TNR']:.4f}")
    print(f"  ARUC: {metrics['ARUC']:.4f}")
    print(f"  mean_test_accuracy: {metrics['mean_test_accuracy']:.4f}")

if __name__ == '__main__':
    main()
