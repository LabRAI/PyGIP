
# ENZYMES Graph Classification — PyGIP-style GNNFingers (PyTorch Geometric)

This repo is split out from a Colab/IPython notebook into a **PyGIP guideline**-style project.

## Folder layout
```
enzymes_gc_pygip/
├─ utils/                       # small helpers (device, seed, meters)
├─ core/                        # base Dataset/Attack/Defense abstractions
├─ datasets/                    # dataset wrappers (PyG)
├─ models/                      # GNN backbones
├─ attacks/                     # training target + suspects (F+/F-)
├─ defenses/                    # fingerprints + verifier + evaluation
└─ examples/                    # runnable example scripts
```
Key modules:
- `datasets/enzymes.py` → `EnzymesDataset` (TUDataset ENZYMES, 70/10/20 split)
- `models/graph_classifiers.py` → `GCN`, `GraphSAGE` (graph classification heads)
- `attacks/gnnfingers_enzymes_gc.py` → trains the **target model**, builds suspects
- `defenses/gnnfingers_enzymes_gc.py` → **FingerprintSet**, **Verifier**, **GNNFingersDefense**, `evaluate()`
- `examples/enzymes_graphclassification.py` → end-to-end runnable example

## Install (CPU-only example)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install 'torch-geometric' 'torch-scatter' 'torch-sparse' 'torch-cluster' -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install scikit-learn numpy
```

## Run
From the project root:
```bash
python -m examples.enzymes_graphclassification
```

## What to upload
Upload the **whole folder** (or the ZIP) to your professor's repo, then open a PR following their guideline.

> Tip: keep notebooks out of the PR (they’re not needed). The code here is enough.

---
**Mapping (Notebook → Project)**
- Dataset → `datasets/enzymes.py`
- Models → `models/graph_classifiers.py`
- Attack → `attacks/gnnfingers_enzymes_gc.py`
- Defense/Fingerprints/Verifier/Eval → `defenses/gnnfingers_enzymes_gc.py`
- Example → `examples/enzymes_graphclassification.py`
```
