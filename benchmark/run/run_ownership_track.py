# run/run_ownership_track.py
import argparse
import json
import os
import traceback
import sys

# Ensure this folder is importable for `utils_benchmark`
sys.path.insert(0, os.path.dirname(__file__))

import torch

from utils_benchmark import (
    load_dataset, instantiate_defense, call_defense,
    write_jsonl, normalize_defense_return, timestamp, ensure_dir
)


def parse_args():
    p = argparse.ArgumentParser(description="GraphIP-Bench: Ownership track (RQ2/RQ3 + grid search).")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--defenses", nargs="+", required=True,
                   help="Keys: randomwm backdoorwm survivewm imperceptiblewm")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--defense_grid_json", type=str, default=None,
                   help="JSON file with per-defense grids of ctor/run kwargs.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--outdir", type=str, default="outputs/RQ2_RQ3")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def load_defense_grid(path: str, defenses):
    if not path:
        return {k: [{"ctor": {}, "run": {}}] for k in defenses}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    grid = {}
    for k in defenses:
        cfgs = raw.get(k, None)
        if not cfgs:
            cfgs = [{"ctor": {}, "run": {}}]
        norm = []
        for item in cfgs:
            if "ctor" in item or "run" in item:
                norm.append({"ctor": item.get("ctor", {}), "run": item.get("run", {})})
            else:
                norm.append({"ctor": item, "run": {}})
        grid[k] = norm
    return grid


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1] if "cuda" in args.device else ""
    ensure_dir(args.outdir)

    header = {
        "runner": "ownership_track",
        "timestamp": timestamp(),
        "datasets": args.datasets,
        "defenses": args.defenses,
        "seeds": args.seeds,
        "device": args.device,
    }
    print(header)

    grid = load_defense_grid(args.defense_grid_json, args.defenses)

    for ds_name in args.datasets:
        dataset = load_dataset(ds_name, root=args.root)
        for defense_key in args.defenses:
            cfg_list = grid[defense_key]
            for cfg_idx, cfg in enumerate(cfg_list):
                ctor_kwargs = cfg.get("ctor", {})
                run_kwargs = cfg.get("run", {})
                for seed in args.seeds:
                    print(f"[RUN] ds={ds_name} defense={defense_key} cfg#{cfg_idx} ctor={ctor_kwargs} run={run_kwargs} seed={seed}")

                    if args.dry_run:
                        continue

                    try:
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        defense = instantiate_defense(defense_key, dataset, ctor_kwargs)
                        perf, comp = normalize_defense_return(call_defense(defense, run_kwargs, seed))

                        record = {
                            "track": "ownership",
                            "dataset": ds_name,
                            "defense": defense_key,
                            "config_index": cfg_idx,
                            "config": cfg,
                            "seed": seed,
                            "perf": perf,
                            "comp": comp,
                        }
                    except Exception as e:
                        record = {
                            "track": "ownership",
                            "dataset": ds_name,
                            "defense": defense_key,
                            "config_index": cfg_idx,
                            "config": cfg,
                            "seed": seed,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }

                    out_jsonl = os.path.join(args.outdir, f"{ds_name}.jsonl")
                    write_jsonl(out_jsonl, record)

    print(f"[DONE] Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
