#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"

DATASETS=("Cora" "Citeseer" "PubMed" "Amazon-Photo" "Proteins")
ATTACKS=("mea0" "mea1" "mea2" "mea3" "mea4" "mea5" "advmea" "cega" "realistic" "dfea_i" "dfea_ii" "dfea_iii")
BUDGETS=(0.25 0.5 1.0 2.0 4.0)
REGIMES=("both" "x_only" "a_only" "data_free")
SEEDS=(0 1 2)

CONF_DIR="configs"
OUTDIR="outputs/RQ1"
LEADER_DIR="outputs/leaderboards"
mkdir -p "${CONF_DIR}" "${OUTDIR}" "${LEADER_DIR}"

GRID_JSON="${CONF_DIR}/attack_grids_large.json"
cat > "${GRID_JSON}" <<'JSON'
{
  "mea0": [ { "ctor": {}, "run": {} } ],
  "mea1": [ { "ctor": {}, "run": {} } ],
  "mea2": [ { "ctor": {}, "run": {} } ],
  "mea3": [ { "ctor": {}, "run": {} } ],
  "mea4": [ { "ctor": {}, "run": {} } ],
  "mea5": [ { "ctor": {}, "run": {} } ],

  "advmea": [ { "ctor": {}, "run": {} } ],

  "cega": [
    { "ctor": {}, "run": {"epochs_per_cycle": 1, "LR_CEGA": 0.01, "setup": "experiment"} },
    { "ctor": {}, "run": {"epochs_per_cycle": 2, "LR_CEGA": 0.01, "setup": "experiment"} },
    { "ctor": {}, "run": {"epochs_per_cycle": 1, "LR_CEGA": 0.005, "setup": "experiment"} },
    { "ctor": {}, "run": {"epochs_per_cycle": 1, "LR_CEGA": 0.01, "setup": "perturbation", "num_perturbations": 100, "noise_level": 0.05} }
  ],

  "realistic": [
    { "ctor": {"hidden_dim": 32,  "threshold_s": 0.60, "threshold_a": 0.40}, "run": {} },
    { "ctor": {"hidden_dim": 64,  "threshold_s": 0.70, "threshold_a": 0.50}, "run": {} },
    { "ctor": {"hidden_dim": 128, "threshold_s": 0.75, "threshold_a": 0.60}, "run": {} },
    { "ctor": {"hidden_dim": 64,  "threshold_s": 0.80, "threshold_a": 0.65}, "run": {} }
  ],

  "dfea_i":   [ { "ctor": {}, "run": {} } ],
  "dfea_ii":  [ { "ctor": {}, "run": {} } ],
  "dfea_iii": [ { "ctor": {}, "run": {} } ]
}
JSON

echo "[RQ1] Sweep attacks (with grids) on GPU ${GPU_ID} ..."
python run/run_attack_track.py \
  --datasets "${DATASETS[@]}" \
  --attacks "${ATTACKS[@]}" \
  --budgets "${BUDGETS[@]}" \
  --regimes "${REGIMES[@]}" \
  --seeds "${SEEDS[@]}" \
  --attack_grid_json "${GRID_JSON}" \
  --device "cuda:${GPU_ID}" \
  --outdir "${OUTDIR}"

echo "[RQ1] Select best configs ..."
python run/select_best.py --rq1_dir "${OUTDIR}" --outdir "${LEADER_DIR}"

echo "[RQ1] Export LaTeX tables ..."
python run/to_latex.py --outdir "${LEADER_DIR}"

echo "[RQ1] Done. Raw -> ${OUTDIR}, Leaderboards & LaTeX -> ${LEADER_DIR}"
