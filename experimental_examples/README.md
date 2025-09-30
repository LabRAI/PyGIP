
# 🔍 GNNFingers Experimentals 

This folder contains experimental scripts of the **GNNFingers** attack and defense pipeline,
integrated with [PyGIP](https://github.com/yushundong/PyGIP).

The purpose of this example is to hold experimental scripts that reproduce and extend fingerprint-based verification of GNN models.

---

## 📂 Structure

```
experimental_examples/
├── cli.py            # Command-line interface (entrypoint)
├── attacker.py       # Attack + defense logic
├── dataset.py        # Dataset loader
├── models.py         # GNN model definitions
├── fingerprints.py   # Fingerprint generation & univerifier
└── README.md         # This documentation
```

---

## ▶️ Usage

All experiments are launched via the CLI:

```bash
python gnnfingers-examples/cli.py --dataset Cora --joint_steps 50
```

### Common options

- `--dataset {Cora,Citeseer,Pubmed}`  
  Which dataset to use.

- `--joint_steps INT`  
  Number of training steps for the joint optimization of fingerprints and univerifier.

- `--num_graphs INT`  
  Number of fingerprint probe graphs.

- `--num_nodes INT`  
  Number of nodes per probe graph.

- `--edge_density FLOAT`  
  Edge density for fingerprint graphs (default 0.05).

- `--proj_every INT`  
  Projection frequency during fingerprint optimization.

- `--node_sample INT`  
  Node sampling factor for graph generation.

- `--device {cpu,cuda}`  
  Device for training (defaults to `cuda` if available).

- `--mode {attack,defense}`  
  Run attack pipeline (default) or defense pipeline.

- `--clean`  
  Remove old `.pt` and `.json` artifacts before running.

---

## 🧪 Examples

### Quick test (small run)
```bash
python experimental_examples/cli.py --dataset Cora --joint_steps 10 --num_graphs 8 --num_nodes 16 --clean
```

### Full attack run
```bash
python experimental_examples/cli.py --dataset Cora --joint_steps 300 --num_graphs 64 --num_nodes 32 --edge_density 0.05
```

### Defense run
```bash
python experimental_examples/cli.py --dataset Cora --mode defense
```

---

## 📦 Outputs

Running the pipeline produces:

- **Model checkpoints (`*.pt`)**  
  - `target_main.pt`, `ft_last.pt`, `reinit_last.pt`, etc.  
- **Fingerprint artifacts**  
  - `fingerprints.pt`, `univerifier.pt`  
- **Verification metrics**  
  - `verification_metrics.json` (contains ROC_AUC, ARUC, robustness, etc.)

---

## 📝 Notes

- The implementation follows the guidelines in `IMPLEMENTATION.md`.  
- The `attack()` and `defense()` functions are public entrypoints, with helpers defined internally.  
- Use the `--clean` flag to avoid piling up old artifacts across runs.

---
