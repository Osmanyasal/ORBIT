#!/usr/bin/env bash
# Snapshot analysis + best-settings generation for one program.
#
# 1. runs the program once in ORBIT Snapshot mode (LD_PRELOAD=liborbit.so, ORBIT_OPTIMIZED_CONF unset),
# 2. feeds the resulting OPTKIT execution folder to orbit-predictor.py generate,
# 3. writes <out>/best_per_region.json holding the predicted `fastest` and `best_edp` settings of every
#    region. Apply it with orbit-optimize.sh.
#
# Usage: orbit-analyze.sh [options] -- <executable> [args...]
#   -o, --out DIR          output directory (default: ./orbit_<executable name>)
#   -m, --model-dir DIR    trained models (model_fastest.pt / model_best_edp.pt)
#                          (default: $ORBIT_MODEL_DIR or <this folder>/models_best)
#   --max-threads N        threads of the target machine (default: nproc)
#   --freq-range MIN MAX   frequency range in kHz of the target machine (default: cpufreq of this node)
#   --freq-step KHZ        frequency step in kHz (default: 200000)
#   --skip-first N         ignore the first N calls of each region
#   --stat mean|median     how repeated calls of a region are compressed
#   --time-source energy|pmu
#   --keep-run DIR         analyse an existing execution folder instead of running the program
#   -h, --help
# Settings of the snapshot run itself are taken from the environment (OMP_NUM_THREADS, OMP_SCHEDULE,
# ORBIT_CPU_FREQ). ORBIT_LIB overrides the liborbit.so path, PYTHON the interpreter used for the predictor
# (it needs torch, pandas and numpy; a working one is searched for otherwise).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_LIB="${ORBIT_LIB:-$HERE/../bin/Release/liborbit.so}"
MODEL_DIR="${ORBIT_MODEL_DIR:-$HERE/models_best}"
OUT=""
EXISTING_RUN=""
PRED_ARGS=()

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "orbit-analyze: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--out) OUT="${2:?missing value for $1}"; shift 2 ;;
        -m|--model-dir) MODEL_DIR="${2:?missing value for $1}"; shift 2 ;;
        --max-threads) PRED_ARGS+=(--max-threads "${2:?missing value for $1}"); shift 2 ;;
        --freq-range) [[ $# -ge 3 ]] || die "--freq-range needs MIN_KHZ MAX_KHZ"
                      PRED_ARGS+=(--freq-range "$2" "$3"); shift 3 ;;
        --freq-step) PRED_ARGS+=(--freq-step "${2:?missing value for $1}"); shift 2 ;;
        --skip-first) PRED_ARGS+=(--skip-first "${2:?missing value for $1}"); shift 2 ;;
        --stat) PRED_ARGS+=(--stat "${2:?missing value for $1}"); shift 2 ;;
        --time-source) PRED_ARGS+=(--time-source "${2:?missing value for $1}"); shift 2 ;;
        --keep-run) EXISTING_RUN="${2:?missing value for $1}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) die "unknown option $1 (see --help)" ;;
        *) break ;;
    esac
done

# --- python with torch/pandas/numpy ------------------------------------------------------------
VENV_SP=("$HERE"/venv-orbit/lib/python*/site-packages)
find_python() {
    local py sp
    local candidates=("${PYTHON:-}" python3.12 python3.11 python3.10 python3.9 python3.8 python3.7 python3 "$HERE/venv-orbit/bin/python")
    for py in "${candidates[@]}"; do
        [[ -n "$py" ]] && command -v "$py" >/dev/null 2>&1 || continue
        for sp in "" "${VENV_SP[@]}"; do
            [[ -z "$sp" || -d "$sp" ]] || continue
            if PYTHONPATH="${sp}${sp:+:}${PYTHONPATH:-}" "$py" -c \
                'import sys; assert sys.version_info >= (3, 7); import torch, pandas, numpy' >/dev/null 2>&1; then
                PY="$py"
                export PYTHONPATH="${sp}${sp:+:}${PYTHONPATH:-}"
                return 0
            fi
        done
    done
    return 1
}

if [[ -z "$EXISTING_RUN" ]]; then
    [[ $# -ge 1 ]] || { usage >&2; exit 1; }
    [[ -f "$ORBIT_LIB" ]] || die "liborbit.so not found at $ORBIT_LIB (build it or set ORBIT_LIB)"
    ORBIT_LIB="$(realpath -- "$ORBIT_LIB")"

    TARGET="$1"
    if [[ "$TARGET" == */* ]]; then
        [[ -x "$TARGET" ]] || die "executable not found or not executable: $TARGET"
    else
        TARGET="$(command -v -- "$TARGET")" || die "executable not found in PATH: $1"
    fi
    TARGET="$(realpath -- "$TARGET")"
    shift
    NAME="$(basename -- "$TARGET")"
fi

[[ -d "$MODEL_DIR" ]] || die "model directory not found: $MODEL_DIR (train with orbit-predictor.py train, or use -m)"
MODEL_DIR="$(realpath -- "$MODEL_DIR")"
find_python || die "no Python >= 3.7 with torch, pandas and numpy found; set PYTHON=/path/to/python"

OUT="${OUT:-$PWD/orbit_${NAME:-analysis}}"
mkdir -p -- "$OUT"
OUT="$(realpath -- "$OUT")"

# --- 1. snapshot run ---------------------------------------------------------------------------
if [[ -n "$EXISTING_RUN" ]]; then
    RUN_DIR="$(realpath -- "$EXISTING_RUN")"
else
    RUN_DIR="$OUT/snapshot_$(date +%Y%m%d_%H%M%S)"
    mkdir -p -- "$RUN_DIR"
    echo "[orbit-analyze] snapshot run of $TARGET $* in $RUN_DIR"
    # OPTKIT creates its execution folder in the working directory.
    (
        cd -- "$RUN_DIR"
        unset ORBIT_OPTIMIZED_CONF ORBIT_OPTIMIZE_POLICY ORBIT_TIME_THRESHOLD
        LD_PRELOAD="$ORBIT_LIB${LD_PRELOAD:+:$LD_PRELOAD}" "$TARGET" "$@"
    ) || die "the program failed in snapshot mode (see output above)"
fi

# a folder holding snapshot.conf, or a folder of such executions, is accepted by the predictor
if [[ ! -f "$RUN_DIR/snapshot.conf" ]] && ! compgen -G "$RUN_DIR/*/snapshot.conf" >/dev/null; then
    die "no snapshot.conf found under $RUN_DIR; did ORBIT intercept an OpenMP region? (is liborbit.so built for this OpenMP runtime?)"
fi

# --- 2. predict fastest / best_edp settings ------------------------------------------------------
CONF="$OUT/best_per_region.json"
echo "[orbit-analyze] generating settings with $PY, models: $MODEL_DIR"
"$PY" "$HERE/orbit-predictor.py" generate \
    --model-dir "$MODEL_DIR" \
    --snapshots "$RUN_DIR" \
    --output "$CONF" \
    "${PRED_ARGS[@]}"

echo
echo "[orbit-analyze] settings written to $CONF"
echo "[orbit-analyze] apply them with:"
echo "    $HERE/orbit-optimize.sh -c $CONF -- ${TARGET:-<executable>} ..."
