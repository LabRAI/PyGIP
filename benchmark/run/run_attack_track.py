# run/run_attack_track.py
import argparse
import json
import os
import traceback
from itertools import product

# Ensure this folder is importable for `utils_benchmark`
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch

from utils_benchmark import (
    load_dataset, compute_fraction_for_budget, instantiate_attack, call_attack,
    write_jsonl, normalize_attack_return, timestamp, ensure_dir
)


def parse_args():
    p = argparse.ArgumentParser(description="GraphIP-Bench: Attack track runner (RQ1 + grid search).")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--attacks", nargs="+", required=True,
                   help="Keys: mea0 mea1 mea2 mea3 mea4 mea5 advmea cega realistic dfea_i dfea_ii dfea_iii")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--regimes", nargs="+", default=["both", "x_only", "a_only", "data_free"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--attack_grid_json", type=str, default=None,
                   help="JSON file with per-attack grids of ctor/run kwargs.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--outdir", type=str, default="outputs/RQ1")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def regime_to_ratios(regime: str, fraction: float):
    if regime == "both":
        return fraction, fraction
    if regime == "x_only":
        return fraction, 0.0
    if regime == "a_only":
        return 0.0, fraction
    if regime == "data_free":
        return 0.0, 0.0
    raise ValueError(f"Unknown regime: {regime}")


def load_attack_grid(path: str, attacks):
    if not path:
        return {k: [{"ctor": {}, "run": {}}] for k in attacks}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    grid = {}
    for k in attacks:
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
        "runner": "attack_track",
        "timestamp": timestamp(),
        "datasets": args.datasets,
        "attacks": args.attacks,
        "budgets": args.budgets,
        "regimes": args.regimes,
        "seeds": args.seeds,
        "device": args.device,
    }
    print(header)

    grid = load_attack_grid(args.attack_grid_json, args.attacks)

    for ds_name in args.datasets:
        dataset = load_dataset(ds_name, root=args.root)
        for budget, regime, seed, attack_key in product(args.budgets, args.regimes, args.seeds, args.attacks):
            fraction = compute_fraction_for_budget(dataset, budget)
            x_ratio, a_ratio = regime_to_ratios(regime, fraction)

            cfg_list = grid[attack_key]
            for cfg_idx, cfg in enumerate(cfg_list):
                ctor_kwargs = cfg.get("ctor", {})
                run_kwargs = cfg.get("run", {})

                print(f"[RUN] ds={ds_name} atk={attack_key} cfg#{cfg_idx} "
                      f"budget={budget} frac={fraction:.5f} regime={regime} "
                      f"x={x_ratio:.5f} a={a_ratio:.5f} seed={seed} ctor={ctor_kwargs} run={run_kwargs}")

                if args.dry_run:
                    continue

                try:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    attack = instantiate_attack(attack_key, dataset, fraction, x_ratio, a_ratio, ctor_kwargs)
                    perf, comp = normalize_attack_return(call_attack(attack, run_kwargs, seed))

                    record = {
                        "track": "attack",
                        "dataset": ds_name,
                        "attack": attack_key,
                        "config_index": cfg_idx,
                        "config": cfg,
                        "budget_mult": budget,
                        "fraction": fraction,
                        "regime": regime,
                        "attack_x_ratio": x_ratio,
                        "attack_a_ratio": a_ratio,
                        "seed": seed,
                        "perf": perf,
                        "comp": comp,
                    }
                except Exception as e:
                    record = {
                        "track": "attack",
                        "dataset": ds_name,
                        "attack": attack_key,
                        "config_index": cfg_idx,
                        "config": cfg,
                        "budget_mult": budget,
                        "fraction": fraction,
                        "regime": regime,
                        "attack_x_ratio": x_ratio,
                        "attack_a_ratio": a_ratio,
                        "seed": seed,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                out_jsonl = os.path.join(args.outdir, f"{ds_name}.jsonl")
                write_jsonl(out_jsonl, record)

    print(f"[DONE] Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
