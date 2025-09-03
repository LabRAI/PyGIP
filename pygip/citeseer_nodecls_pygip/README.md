# CiteSeer Node Classification · PyGIP-Ready (GNNFingers)

This package refactors a Colab notebook into a **PyGIP-compliant** modular repo for **CiteSeer (node classification)**.

## ✅ What’s inside
```
citeseer_nodecls_pygip/
├─ utils/common.py
├─ datasets/citeseer.py
├─ models/node_classifiers.py
├─ attacks/gnnfingers_citeseer_nc.py
├─ defenses/gnnfingers_citeseer_nc.py
└─ examples/citeseer_nodeclassification.py
```

- Uses `api_type='pyg'` and `torch_geometric.data.Data` for `graph_data`.
- `BaseAttack` / `BaseDefense` manage `self.device` internally (no manual overrides in subclasses).
- `GNNFingersAttack.attack()` returns dict with `target_model`, `positive_models`, `negative_models`.
- `GNNFingersDefense.defend()` trains fingerprints + verifier; `evaluate()` reports **TPR/TNR/ARUC/Test Acc**.
- `examples/` has a runnable script for quick review.

## 📦 Install (same as notebook)
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install scikit-learn numpy
```

## ▶️ Run example
From the project root:
```bash
python -m examples.citeseer_nodeclassification
```

## 🔁 Where to plug in PyGIP
- Drop this folder into your fork at an appropriate location (e.g. `contrib/gnnfingers_citeseer_nodecls/`).
- Ensure imports resolve within the package context (relative or module-level as above).
- Provide this `examples/` script in your PR to **speed up code review** (per guideline).

## Notes
- Default split is a random 70/10/20 node split (seed=7). Change inside `datasets/citeseer.py` if needed.
- GNN backbones: `GCNNodeClassifier`, `SAGENodeClassifier` (extensible).
- Fingerprint dimension: `64 nodes × num_classes (6) = 384` → verifier MLP input.
