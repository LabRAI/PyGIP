
# PROTEINS Graph Classification — PyGIP-style GNNFingers (Attack + Defense)

This folder is a PyGIP-compliant refactor of your Colab notebook for **PROTEINS (graph classification)**.

## Layout
```
utils/common.py                    # get_device, set_seed, AvgMeter
core/base.py                       # Dataset, BaseAttack, BaseDefense
datasets/proteins.py               # ProteinsDataset (api_type='pyg', graph_data is a Data instance)
models/graph_classifiers.py        # GCN, GraphSAGE graph classifiers
attacks/gnnfingers_proteins_gc.py  # GNNFingersAttack (builds victim + suspects)
defenses/gnnfingers_proteins_gc.py # Fingerprints, Verifier, GNNFingersDefense, evaluate()
examples/proteins_graphclassification.py  # runnable script
```

## Run locally
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install scikit-learn numpy

python -m examples.proteins_graphclassification
```

## PyGIP requirements satisfied
- `api_type='pyg'` and `graph_data` is a `torch_geometric.data.Data` → ✅
- `BaseAttack`/`BaseDefense` set `self.device` internally → ✅
- `attack()` returns a dict: `target_model`, `positive_models`, `negative_models` → ✅
- `defend()` trains fingerprints + verifier; example script in `examples/` → ✅
