# Cora Link Prediction — GNNFingers (PyGIP-style)

Modular implementation for **Cora (Link Prediction)** with **GNNFingers attack + defense** following the PyGIP guideline.

## Project Layout
```
cora_linkpred_pygip/
├─ datasets/
│  └─ cora.py
├─ models/
│  └─ link_predictors.py
├─ attacks/
│  └─ gnnfingers_cora.py
├─ defenses/
│  └─ gnnfingers_cora.py
└─ examples/
   └─ cora_linkprediction.py
```

## Setup
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install scikit-learn numpy
```

## Run
```bash
python -m examples.cora_linkprediction
```

## Notes
- Dataset uses `api_type='pyg'` and maintains `graph_data` as `torch_geometric.data.Data`.
- Device handling is via `get_device()` in attack/defense modules (aligned with BaseAttack/BaseDefense convention).
- Example trains a victim model, builds suspect models, learns fingerprints, and evaluates ownership verification.
- Tune `joint_iters`, `k_pos/k_neg`, and hidden sizes for speed/quality trade-offs.
Generated: 2025-09-03T20:58:59
