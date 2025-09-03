
# GNNFingers (AIDS) — PyGIP-Style Layout

This repo splits your original single script into packages:

```
gnnfingers_pygip/
  core/                # BaseAttack, BaseDefense
  datasets/            # AIDSMatchingDataset
  models/              # GCN, GraphSAGE backbones
  attacks/             # GNNFingersAttack (AIDS)
  defenses/            # GNNFingersDefense (AIDS)
  fingerprints/        # FingerprintGraphPair, FingerprintSet, Verifier
  eval.py              # evaluate()
  examples/
    run_gnnfingers_aids.py
```

Run example:
```bash
python -m gnnfingers_pygip.examples.run_gnnfingers_aids
```
