#!/usr/bin/env python3
# Quick check tool to report surrogate checkpoint metadata.
import argparse, torch, os
p = argparse.ArgumentParser()
p.add_argument("surrogate", help="path to surrogate checkpoint (pth)")
args = p.parse_args()
if not os.path.exists(args.surrogate):
    print("surrogate not found:", args.surrogate); raise SystemExit(2)
ck = torch.load(args.surrogate, map_location="cpu")
if isinstance(ck, dict):
    print("keys:", list(ck.keys()))
    for k in ("dataset_name","num_features"):
        if k in ck: print(k, "=", ck[k])
else:
    print("raw object type:", type(ck))
