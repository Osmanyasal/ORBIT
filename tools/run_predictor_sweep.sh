#!/usr/bin/env bash
# Network/optimisation sweep for orbit-predictor.py with leave-one-application-out CV (nothing is saved).
# Usage: ./run_predictor_sweep.sh [WORKERS]   (default: all cores of the node)
# Runs 10 variants (the default setting is "base") x 2 seeds x 2 targets (22 folds each), one single-threaded
# training per worker. At the end of each target the variant with the lowest mean geomean loss is selected
# and the final model is trained on all regions: ../../models_best/model_fastest.pt and model_best_edp.pt
# (+ best_<target>.json with the chosen variant). Training data: ../../P12-data (override: ORBIT_DATA_DIR).
# Results: predictor_sweep_logs/<target>_app_<variant>_s<seed>.log and predictor_sweep_logs/results_*_app.txt
# The virtualenv ORBIT/tools/venv-orbit is created (and activated) if it is missing or unusable on this node.
# TORCH_INDEX overrides the pip index used for torch (default: CPU wheels, the code does not use a GPU).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv-orbit"
WORKERS="${1:-$(nproc)}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

if ! "$VENV/bin/python" -c "import torch, pandas, numpy" >/dev/null 2>&1; then
    echo "venv missing or unusable on $(hostname -s): creating $VENV"
    python3 -m venv --clear "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch --index-url "$TORCH_INDEX"
    "$VENV/bin/pip" install numpy pandas
fi
source "$VENV/bin/activate"

cd "$HERE"
echo "python: $(command -v python)  workers: $WORKERS"

# leave-one-application-out cross-validation (estimate for unseen applications), both targets
mkdir -p predictor_sweep_logs
python -u predictor_sweep.py fastest  app "$WORKERS" 2>&1 | tee predictor_sweep_logs/results_fastest_app.txt
python -u predictor_sweep.py best_edp app "$WORKERS" 2>&1 | tee predictor_sweep_logs/results_best_edp_app.txt
