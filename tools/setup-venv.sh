#!/usr/bin/env bash
# Creates (or repairs) the virtual environment used by orbit-predictor.py: tools/venv-orbit with torch
# (CPU build; the predictor does not use a GPU), numpy and pandas.
# Usage: setup-venv.sh [--force]
#   --force   recreate the environment even if the current one works
# Environment: PYTHON (interpreter used to create it, default: newest python3.9-3.13 found),
#              VENV (target directory, default: <this folder>/venv-orbit),
#              TORCH_INDEX (pip index for torch, default: https://download.pytorch.org/whl/cpu).
# The environment is tied to the node's Python: if tools/venv-orbit was created on another node and
# its interpreter does not work here, run this script again on the node you use.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/venv-orbit}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
FORCE=0
[[ "${1:-}" != "--force" ]] || FORCE=1

if [[ "$FORCE" -eq 0 ]] && "$VENV/bin/python" -c \
    'import sys; assert sys.version_info >= (3, 7); import torch, pandas, numpy' >/dev/null 2>&1; then
    echo "[setup-venv] $VENV already works on $(hostname -s) ($("$VENV/bin/python" --version 2>&1))"
    exit 0
fi

PY=""
for cand in "${PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    [[ -n "$cand" ]] && command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys; assert sys.version_info >= (3, 9)' >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
[[ -n "$PY" ]] || { echo "[setup-venv] no Python >= 3.9 found on $(hostname -s); load one (module load Python) or set PYTHON=" >&2; exit 1; }

echo "[setup-venv] creating $VENV with $("$PY" --version 2>&1) on $(hostname -s)"
"$PY" -m venv --clear "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch --index-url "$TORCH_INDEX"
"$VENV/bin/pip" install numpy pandas
"$VENV/bin/python" -c 'import torch, pandas, numpy; print("[setup-venv] ok, torch", torch.__version__)'
