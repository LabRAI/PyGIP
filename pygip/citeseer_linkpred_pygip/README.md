# Citeseer Link Prediction — GNNFingers (PyGIP-style)

Modular, guideline-compliant implementation for **Citeseer (Link Prediction)** with **GNNFingers attack + defense**.

## Layout
```
citeseer_linkpred_pygip/
├─ utils/
│  └─ common.py                    # get_device, set_seed
├─ datasets/
│  └─ citeseer.py                  # Dataset wrapper (RandomLinkSplit; PyG Data)
├─ models/
│  └─ link_predictors.py           # GCN/SAGE encoders for LP
├─ attacks/
│  └─ gnnfingers_citeseer_lp.py    # Victim + F+/F- suspect generation
├─ defenses/
│  └─ gnnfingers_citeseer_lp.py    # Fingerprints + Verifier + Defense + evaluate()
└─ examples/
   └─ citeseer_linkprediction.py   # Runnable example
```

## Setup
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install scikit-learn numpy
```

## Run
```bash
python -m examples.citeseer_linkprediction
```

## Notes
- Uses `api_type='pyg'` and keeps `graph_data` as `torch_geometric.data.Data`.
- Device handled once via `get_device()` in base classes; subclasses don't override device logic.
- Attack implements `_train_target_model()`, multiple positive/negative suspect builders, and `attack()` returns pools.
- Defense performs joint training of fingerprint features and verifier, then reports Robustness/Uniqueness/ARUC.

Generated: 2025-09-03T21:12:07
