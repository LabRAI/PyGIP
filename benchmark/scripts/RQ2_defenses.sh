#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"

DATASETS=("Cora" "Citeseer" "PubMed" "Amazon-Photo" "Proteins")
DEFENSES=("randomwm" "backdoorwm" "survivewm" "imperceptiblewm")
SEEDS=(0 1 2)

CONF_DIR="configs"
OUTDIR="outputs/RQ2_RQ3"
LEADER_DIR="outputs/leaderboards"
mkdir -p "${CONF_DIR}" "${OUTDIR}" "${LEADER_DIR}"

GRID_JSON="${CONF_DIR}/defense_grids_large.json"
cat > "${GRID_JSON}" <<'JSON'
{
  "randomwm": [
    {"ctor": {"wm_ratio": 0.002}},
    {"ctor": {"wm_ratio": 0.005}},
    {"ctor": {"wm_ratio": 0.010}},
    {"ctor": {"wm_ratio": 0.020}},
    {"ctor": {"wm_ratio": 0.050}}
  ],
  "backdoorwm": [
    {"ctor": {"trigger_density": 0.01}},
    {"ctor": {"trigger_density": 0.02}},
    {"ctor": {"trigger_density": 0.05}},
    {"ctor": {"trigger_density": 0.10}},
    {"ctor": {"trigger_density": 0.20}}
  ],
  "survivewm": [
    {"ctor": {"wm_strength": 0.25}},
    {"ctor": {"wm_strength": 0.50}},
    {"ctor": {"wm_strength": 1.00}},
    {"ctor": {"wm_strength": 1.50}},
    {"ctor": {"wm_strength": 2.00}}
  ],
  "imperceptiblewm": [
    {"ctor": {"epsilon": 0.25}},
    {"ctor": {"epsilon": 0.50}},
    {"ctor": {"epsilon": 1.00}},
    {"ctor": {"epsilon": 2.00}},
    {"ctor": {"epsilon": 4.00}}
  ]
}
JSON

echo "[RQ2] Sweep defenses on GPU ${GPU_ID} ..."
python run/run_ownership_track.py \
  --datasets "${DATASETS[@]}" \
  --defenses "${DEFENSES[@]}" \
  --defense_grid_json "${GRID_JSON}" \
  --seeds "${SEEDS[@]}" \
  --device "cuda:${GPU_ID}" \
  --outdir "${OUTDIR}"

echo "[RQ2] Select best configs ..."
python run/select_best.py --rq2_dir "${OUTDIR}" --outdir "${LEADER_DIR}"

echo "[RQ2] Export LaTeX tables ..."
python run/to_latex.py --outdir "${LEADER_DIR}"

echo "[RQ2] Done. Raw -> ${OUTDIR}, Leaderboards & LaTeX -> ${LEADER_DIR}"
