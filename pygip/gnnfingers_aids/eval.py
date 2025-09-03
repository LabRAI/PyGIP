
import torch, torch.nn.functional as F
from .utils import AvgMeter

def evaluate(defense, pos_models, neg_models):
    print("Evaluating...")
    robustness_TPR = []; uniqueness_TNR = []
    for pos_m in pos_models:
        res = defense.verify_ownership(pos_m)
        robustness_TPR.append(res['is_stolen'])
    for neg_m in neg_models:
        res = defense.verify_ownership(neg_m)
        uniqueness_TNR.append(1 - res['is_stolen'])
    tpr = sum(robustness_TPR) / len(robustness_TPR) if robustness_TPR else 0
    tnr = sum(uniqueness_TNR) / len(uniqueness_TNR) if uniqueness_TNR else 0
    aruc = (tpr + tnr) / 2

    # 'accuracy-like' from MSE on test pairs
    target_model = getattr(defense, 'target_model', None)
    if target_model is None and pos_models:
        target_model = pos_models[0]
    target_model.eval()
    mse = AvgMeter()
    with torch.no_grad():
        for (g1, g2), sim in zip(defense.dataset.test_pairs, defense.dataset.test_sims):
            g1 = g1.to(defense.device); g2 = g2.to(defense.device)
            pred, _, _ = target_model.forward_pair(g1, g2)
            loss = F.mse_loss(pred, torch.tensor(sim, device=defense.device).view(-1))
            mse.add(loss.item())
    mean_test_accuracy = 1.0 - mse.avg
    return {'robustness_TPR': tpr,'uniqueness_TNR': tnr,'ARUC': aruc,'mean_test_accuracy': mean_test_accuracy}
