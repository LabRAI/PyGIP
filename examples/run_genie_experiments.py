#!/usr/bin/env python3
# CI-friendly wrapper example: run genie extraction (high-level wrapper) and optionally pruning.
# Robust fallbacks for minimal environments.

import argparse, json, os, sys
from typing import Any

# ensure repo root on path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

# high-level extraction wrapper fallback
try:
    from pygip.experiments.model_extraction import run_model_extraction
except Exception:
    def run_model_extraction(dataset_name: str, query_ratio: float, model_path: str = None, **kw):
        print("[fallback] pygip.experiments.model_extraction not available; using tiny fallback run_model_extraction()")
        return {
            "dataset": dataset_name,
            "query_ratio": float(query_ratio),
            "model_path": model_path,
            "status": "fallback",
            "surrogate_path": None,
            "notes": "fallback run_model_extraction - no extraction performed",
        }

# pruning helper fallback
try:
    from experiments.pruning_utils import prune_and_eval
except Exception:
    def prune_and_eval(model, graph, ratio=0.2, device="cpu"):
        print("[fallback] experiments.pruning_utils.prune_and_eval not available; skipping pruning evaluation")
        return None

# dataset loader fallback
try:
    from pygip.datasets.utils import load_dataset
except Exception:
    def load_dataset(name: str):
        raise RuntimeError("pygip.datasets.utils.load_dataset not available in this environment")

# model fallback (minimal graph link predictor)
try:
    from pygip.models.gcn_link_predictor import GCNLinkPredictor
except Exception:
    import torch
    class GCNLinkPredictor(torch.nn.Module):
        def __init__(self, in_channels=64, hidden_channels=64):
            super().__init__()
            self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        def encode(self, x, edge_index=None):
            h = self.lin1(x); h = torch.relu(h); return self.lin2(h)
        def decode(self, z, edge_index):
            if isinstance(edge_index, (list, tuple)): src, dst = edge_index
            else: src, dst = edge_index[0], edge_index[1]
            return (z[src] * z[dst]).sum(dim=1)

# checkpoint loader
def _load_checkpoint(path: str):
    import torch
    ck = torch.load(path, map_location="cpu")
    for k in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(ck, dict) and k in ck:
            return ck[k]
    if isinstance(ck, dict) and all(hasattr(v, "ndim") for v in ck.values()):
        return ck
    return ck

# build model robustly
def _build_model_for_data(data):
    import torch
    in_ch = None
    try:
        if hasattr(data, "x") and getattr(data, "x") is not None:
            in_ch = int(data.x.size(1))
    except Exception:
        in_ch = None
    if in_ch is None:
        try: in_ch = int(getattr(data, "num_features", 64))
        except Exception: in_ch = 64

    try:
        model = GCNLinkPredictor(in_channels=int(in_ch), hidden_channels=64)
        return model
    except TypeError:
        try:
            model = GCNLinkPredictor(int(in_ch), 64)
            return model
        except Exception:
            pass
    except Exception:
        pass

    # fallback tiny model
    class TinyFallback(torch.nn.Module):
        def __init__(self, in_ch=int(in_ch), hidden=64):
            super().__init__()
            self.lin1 = torch.nn.Linear(in_ch, hidden)
            self.lin2 = torch.nn.Linear(hidden, hidden)
        def encode(self, x, edge_index=None):
            h = self.lin1(x); h = torch.relu(h); return self.lin2(h)
        def decode(self, z, edge_index):
            if isinstance(edge_index, (list, tuple)): src, dst = edge_index
            else: src, dst = edge_index[0], edge_index[1]
            return (z[src] * z[dst]).sum(dim=1)

    print("[fallback] using TinyFallback model as GCNLinkPredictor could not be instantiated")
    return TinyFallback(in_ch, 64)

def main():
    parser = argparse.ArgumentParser(description="Run GENIE demo: extraction + pruning (CI-friendly)")
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--model_path", "-m", default=None)
    parser.add_argument("--query_ratio", "-q", type=float, default=0.05)
    parser.add_argument("--prune_ratios", "-p", type=float, nargs="+", default=[0.2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("[GenieModelExtraction] Running extraction (wrapper or fallback)")
    extraction_out = run_model_extraction(dataset_name=args.dataset, query_ratio=args.query_ratio, model_path=args.model_path)
    print("Extraction result:", extraction_out)

    try:
        ds = load_dataset(args.dataset)
    except Exception as e:
        print("[warning] Could not load dataset for pruning:", e)
        ds = None

    graph = getattr(ds, "graph_data", None) or getattr(ds, "data", None) or ds
    model = _build_model_for_data(ds if graph is None else graph)

    if args.model_path:
        try:
            state = _load_checkpoint(args.model_path)
        except Exception as e:
            print("[warning] checkpoint load failed:", e); state = None
        if isinstance(state, dict):
            try:
                model.load_state_dict(state); print("[GenieModelExtraction] Loaded model checkpoint into model.")
            except Exception:
                if isinstance(state, dict) and "model_state" in state:
                    try: model.load_state_dict(state["model_state"])
                    except Exception: print("[warning] Could not load nested state_dict keys; continuing with fresh model.")
        else:
            print("[GenieModelExtraction] Checkpoint loader returned non-dict; continuing with fresh model.")
    model.eval()

    for ratio in args.prune_ratios:
        print(f"[GeniePruning] Running pruning eval (ratio={ratio})")
        try:
            prune_auc = prune_and_eval(model, graph, ratio=ratio, device=args.device)
        except Exception as e:
            print("[warning] prune_and_eval raised an exception:", e); prune_auc = None

        prune_result = {
            "dataset": args.dataset,
            "prune_ratio": ratio,
            "test_auc": float(prune_auc) if prune_auc is not None else None,
            "watermark_auc": None,
        }
        print("Pruning result:", prune_result)
        try:
            print(json.dumps(prune_result))
        except Exception:
            pass

if __name__ == "__main__":
    main()
