\# GNNFingers Unified Pipeline (Example)



End-to-end example to reproduce fingerprint-based ownership verification:

\- Node/Graph classification, Link prediction, Graph matching

\- AUROC and ARUC

\- Optional pairwise verifier (“matcher”)

\- Saves plots/CSV/JSON into `--outdir` (defaults to `outputs/`, which is gitignored)



\## How to run



CPU quick test (small run):

```bash

python examples/gnnfingers/gnnfingers\_full\_pipeline.py --dataset Cora --variants 20 --save-plots



