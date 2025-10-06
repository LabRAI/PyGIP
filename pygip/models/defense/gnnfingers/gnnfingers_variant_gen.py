import torch
import numpy as np

# ----------------------
# Generate variants
# ----------------------
def generate_variants(model, data, task="node", num_variants=20):
    positives, negatives = [], []

    with torch.no_grad():
        if task == "node":
            base_out = model(data.x, data.edge_index)
        else:
            base_out = model(data.x, data.edge_index, data.batch)
        base_fp = base_out.cpu().numpy()
        positives.append(base_fp)

    # Simulate positive variants (small noise/fine-tuning)
    for _ in range(num_variants):
        noise = np.random.normal(0, 0.01, base_fp.shape)
        positives.append(base_fp + noise)

    # Simulate negative variants (label shuffling)
    for _ in range(num_variants):
        neg = np.random.permutation(base_fp)
        negatives.append(neg)

    X = np.vstack(positives + negatives)
    y = np.array([1] * len(positives) + [0] * len(negatives))
    return X, y
