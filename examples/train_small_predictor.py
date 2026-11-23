#!/usr/bin/env python3
# Minimal demo trainer that creates a tiny checkpoint for CI/testing.
import argparse, os
def save_demo_checkpoint(path="examples/watermarked_model_demo.pth"):
    import torch
    from pygip.models.gcn_link_predictor import GCNLinkPredictor
    model = GCNLinkPredictor(in_channels=64, hidden_channels=64)
    ckpt = {"model_state": model.state_dict(), "num_features": 64, "dataset_name": "Cora"}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)
    print("Saved demo teacher checkpoint to", path)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="examples/watermarked_model_demo.pth")
    args = p.parse_args()
    save_demo_checkpoint(args.out)
