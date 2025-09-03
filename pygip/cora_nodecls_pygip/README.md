# Cora Node Classification — GNNFingers (PyGIP-style)

Modular implementation for **Cora (Node Classification)** with **GNNFingers attack + defense** following the PyGIP guideline.

## Project Layout
```
cora_nodecls_pygip/
├─ datasets/
│  └─ cora.py
├─ models/
│  └─ node_classifiers.py
├─ attacks/
│  └─ gnnfingers_cora_node.py
├─ defenses/
│  └─ gnnfingers_cora_node.py
└─ examples/
   └─ cora_nodeclassification.py
```

## Setup
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install scikit-learn numpy
```

## Run
```bash
python -m examples.cora_nodeclassification
```

## Notes
- Uses `api_type='pyg'` and keeps `graph_data` a `torch_geometric.data.Data` object per guideline.
- `BaseAttack`/`BaseDefense` keep device handling consistent (`get_device()`).
- Example trains a Cora GCN victim, creates suspects (F+/F-), learns fingerprints, and evaluates verification metrics.
Generated: 2025-09-03T21:06:03
