# run/utils_benchmark.py
import importlib
import inspect
import json
import os
import sys
import time
from typing import Any, Dict, Tuple

# Ensure repository root is on sys.path so imports like `pygip.*` always work
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


def _try_import(path: str):
    mod, cls = path.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, cls)


# ==== Registries (paths match your repo layout) ====
ATTACK_REGISTRY = {
    # MEA.py — six attacks (0..5)
    "mea0": "pygip.models.attack.mea.MEA.ModelExtractionAttack0",
    "mea1": "pygip.models.attack.mea.MEA.ModelExtractionAttack1",
    "mea2": "pygip.models.attack.mea.MEA.ModelExtractionAttack2",
    "mea3": "pygip.models.attack.mea.MEA.ModelExtractionAttack3",
    "mea4": "pygip.models.attack.mea.MEA.ModelExtractionAttack4",
    "mea5": "pygip.models.attack.mea.MEA.ModelExtractionAttack5",

    # Other attacks
    "advmea": "pygip.models.attack.mea.AdvMEA.AdvMEA",
    "cega": "pygip.models.attack.mea.CEGA.CEGA",
    "realistic": "pygip.models.attack.mea.Realistic.RealisticAttack",
    "dfea_i": "pygip.models.attack.mea.DataFreeMEA.DFEATypeI",
    "dfea_ii": "pygip.models.attack.mea.DataFreeMEA.DFEATypeII",
    "dfea_iii": "pygip.models.attack.mea.DataFreeMEA.DFEATypeIII",
}

DEFENSE_REGISTRY = {
    "randomwm": "pygip.models.defense.atom.RandomWM.RandomWM",
    "backdoorwm": "pygip.models.defense.atom.BackdoorWM.BackdoorWM",
    "survivewm": "pygip.models.defense.atom.SurviveWM.SurviveWM",
    "imperceptiblewm": "pygip.models.defense.atom.ImperceptibleWM.ImperceptibleWM",
    # Add more if needed (e.g., GROVE) using their import paths.
}


def load_dataset(name: str, root: str = None):
    """Robust loader that tries several known entry points."""
    tried = []
    for mod, fn in [
        ("pygip.datasets.datasets", "get_dataset"),
        ("pygip.datasets", "get_dataset"),
        ("pygip.data", "get_dataset"),
        ("pygip.data", "build_dataset"),
    ]:
        try:
            m = importlib.import_module(mod)
            f = getattr(m, fn)
            if root is None:
                ds = f(name)
            else:
                try:
                    ds = f(name, root=root)
                except TypeError:
                    ds = f(name)
            return ds
        except Exception as e:
            tried.append(f"{mod}.{fn}: {e}")
    raise RuntimeError("Could not load dataset. Tried: " + " | ".join(tried))


def get_masks(dataset):
    g = dataset.graph_data
    n = g.number_of_nodes() if hasattr(g, "number_of_nodes") else dataset.num_nodes
    nd = g.ndata
    test_mask = nd["test_mask"]
    train_mask = nd.get("train_mask", torch.zeros(n, dtype=torch.bool))
    val_mask = nd.get("val_mask", torch.zeros(n, dtype=torch.bool))
    return train_mask, val_mask, test_mask


def get_nums(dataset) -> Tuple[int, int, int]:
    n = getattr(dataset, "num_nodes", None) or dataset.graph_data.number_of_nodes()
    d = getattr(dataset, "num_features", None) or dataset.graph_data.ndata["feat"].shape[1]
    c = getattr(dataset, "num_classes", None) or int(dataset.graph_data.ndata["label"].max().item() + 1)
    return n, d, c


def test_size(dataset) -> int:
    _, _, tmask = get_masks(dataset)
    return int(tmask.sum().item())


def compute_fraction_for_budget(dataset, budget_mult: float) -> float:
    """Convert a test-size–relative budget multiplier to a node fraction."""
    tsz = test_size(dataset)
    n, _, _ = get_nums(dataset)
    queries = max(1, int(round(budget_mult * tsz)))
    return min(0.99, queries / float(n))


def _has_var_kw(fn):
    try:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False
    except (TypeError, ValueError):
        return False


def _filter_kwargs(fn, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by `fn` unless it provides **kwargs."""
    if not cfg:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    if _has_var_kw(fn):
        return dict(cfg)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in valid}


def instantiate_attack(key: str, dataset, fraction: float, x_ratio: float, a_ratio: float, ctor_kwargs: Dict[str, Any]):
    cls = _try_import(ATTACK_REGISTRY[key])
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if "attack_x_ratio" in params and "attack_a_ratio" in params:
        safe_ctor = _filter_kwargs(cls.__init__, ctor_kwargs)
        return cls(dataset=dataset, attack_x_ratio=x_ratio, attack_a_ratio=a_ratio, **safe_ctor)
    if "attack_node_fraction" in params:
        safe_ctor = _filter_kwargs(cls.__init__, ctor_kwargs)
        return cls(dataset=dataset, attack_node_fraction=fraction, **safe_ctor)
    if "attack_ratio" in params:
        safe_ctor = _filter_kwargs(cls.__init__, ctor_kwargs)
        return cls(dataset=dataset, attack_ratio=fraction, **safe_ctor)
    safe_ctor = _filter_kwargs(cls.__init__, ctor_kwargs)
    return cls(dataset=dataset, attack_node_fraction=fraction, **safe_ctor)


def call_attack(attack_obj, run_kwargs: Dict[str, Any], seed: int):
    if not hasattr(attack_obj, "attack"):
        raise RuntimeError("Attack object has no `.attack()` method.")
    fn = getattr(attack_obj, "attack")
    safe_run = _filter_kwargs(fn, dict(run_kwargs or {}))
    safe_run["seed"] = seed
    return fn(**safe_run)


def instantiate_defense(key: str, dataset, ctor_kwargs: Dict[str, Any]):
    cls = _try_import(DEFENSE_REGISTRY[key])
    safe_ctor = _filter_kwargs(cls.__init__, ctor_kwargs or {})
    return cls(dataset=dataset, **safe_ctor)


def call_defense(defense_obj, run_kwargs: Dict[str, Any], seed: int):
    for m in ["defend", "run", "fit", "train"]:
        if hasattr(defense_obj, m) and callable(getattr(defense_obj, m)):
            fn = getattr(defense_obj, m)
            safe_run = _filter_kwargs(fn, dict(run_kwargs or {}))
            safe_run["seed"] = seed
            return fn(**safe_run)
    if hasattr(defense_obj, "__call__"):
        fn = getattr(defense_obj, "__call__")
        safe_run = _filter_kwargs(fn, dict(run_kwargs or {}))
        safe_run["seed"] = seed
        return fn(**safe_run)
    raise RuntimeError("No callable entrypoint found on defense object.")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, record: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def normalize_attack_return(ret) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], dict) and isinstance(ret[1], dict):
        return ret[0], ret[1]
    if isinstance(ret, dict):
        return ret, {}
    raise RuntimeError("attack.attack() did not return the expected (perf_dict, comp_dict) or dict.")


def normalize_defense_return(ret) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], dict) and isinstance(ret[1], dict):
        return ret[0], ret[1]
    if isinstance(ret, dict):
        return ret, {}
    raise RuntimeError("defense call did not return the expected (perf_dict, comp_dict) or dict.")


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
