# examples/cora_linkprediction.py
# Runnable example for Cora Link Prediction with GNNFingers (PyGIP-style)

import torch
from sklearn.metrics import average_precision_score

from datasets.cora import CoraDataset
from attacks.gnnfingers_cora import GNNFingersAttack, get_device
from defenses.gnnfingers_cora import GNNFingersDefense

def evaluate(defense, pos_models, neg_models):
    print("Evaluating...")
    robustness_TPR = []
    uniqueness_TNR = []

    # Evaluate positive models
    for pos_m in pos_models:
        res = defense.verify_ownership(pos_m)
        robustness_TPR.append(res['is_stolen'])

    # Evaluate negative models
    for neg_m in neg_models:
        res = defense.verify_ownership(neg_m)
        uniqueness_TNR.append(1 - res['is_stolen'])

    # Calculate metrics
    tpr = sum(robustness_TPR) / len(robustness_TPR) if robustness_TPR else 0
    tnr = sum(uniqueness_TNR) / len(uniqueness_TNR) if uniqueness_TNR else 0
    aruc = (tpr + tnr) / 2

    # Mean Test Accuracy (AUPRC for link prediction)
    target_model = defense.target_model if hasattr(defense, 'target_model') and defense.target_model is not None else pos_models[0]
    target_model.eval()
    with torch.no_grad():
        data = defense.dataset.graph_data.to(defense.device)
        pos_pred, neg_pred, _ = target_model(data)

        # Get test predictions
        test_pos_pred = pos_pred[data.test_mask]
        test_neg_pred = neg_pred[data.neg_test_mask]

        test_labels = torch.cat([
            torch.ones_like(test_pos_pred),
            torch.zeros_like(test_neg_pred)
        ])
        test_preds = torch.cat([test_pos_pred, test_neg_pred])
        test_probs = torch.sigmoid(test_preds)

        auprc = average_precision_score(test_labels.cpu(), test_probs.cpu())

    return {
        'robustness_TPR': tpr,
        'uniqueness_TNR': tnr,
        'ARUC': aruc,
        'mean_test_accuracy': auprc
    }

def main():
    print("🚀 Starting PyGIP-compliant GNNFingers on Cora (Link Prediction)")
    device = get_device()

    # Dataset
    ds = CoraDataset(api_type='pyg', path='./data')
    print(f"Dataset: {ds.get_name()} | nodes={ds.num_nodes} | feat={ds.num_features}")

    # Attack (build victim + suspects)
    attack = GNNFingersAttack(ds, attack_node_fraction=0.1, device=device, victim_hidden=64, victim_out=64)
    atk_res = attack.attack()

    # Defense (joint training)
    defense = GNNFingersDefense(ds, attack_node_fraction=0.1, device=device,
                                fp_nodes=100, backbone_out=64, joint_iters=150,
                                verif_lr=5e-4, fp_lr=2e-3)
    _ = defense.defend(attack_results=atk_res)

    # Quick sanity check
    ex_pos = defense.verify_ownership(atk_res['positive_models'][0])
    ex_neg = defense.verify_ownership(atk_res['negative_models'][0])
    print("\n🔎 Example verification")
    print("Positive model:", ex_pos)
    print("Negative model:", ex_neg)

    # Full eval
    print("\n📈 Evaluating across all suspects...")
    metrics = evaluate(defense, atk_res['positive_models'], atk_res['negative_models'])
    print("Results:")
    print(f"  robustness_TPR: {metrics['robustness_TPR']:.4f}")
    print(f"  uniqueness_TNR: {metrics['uniqueness_TNR']:.4f}")
    print(f"  ARUC: {metrics['ARUC']:.4f}")
    print(f"  mean_test_accuracy: {metrics['mean_test_accuracy']:.4f}")

if __name__ == '__main__':
    main()
