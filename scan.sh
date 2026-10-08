#!/usr/bin/env bash
set -u -o pipefail

# Check arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <executable> [args...]" >&2
    echo "Example: $0 ./your_application arg1 arg2" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_LIB="${ORBIT_LIB:-${SCRIPT_DIR}/bin/Release/liborbit.so}"

if [[ ! -f "${ORBIT_LIB}" ]]; then
    echo "Error: ORBIT library not found at ${ORBIT_LIB}" >&2
    echo "Please build ORBIT first: make -C build config=release orbit" >&2
    exit 1
fi

# Resolve paths before changing directory, since the sweep runs inside a per-application directory.
ORBIT_LIB="$(realpath -- "${ORBIT_LIB}")"
TARGET="$1"
TARGET_NAME="$(basename -- "${TARGET}")"
if [[ "${TARGET}" == */* ]]; then
    if [[ ! -x "${TARGET}" ]]; then
        echo "Error: executable not found or not executable: ${TARGET}" >&2
        exit 1
    fi
    TARGET="$(realpath -- "${TARGET}")"
fi
set -- "${TARGET}" "${@:2}"

# Run in <current directory>/<application name>, so OPTKIT's output folders are grouped per application.
WORK_DIR="${PWD}/${TARGET_NAME}"
mkdir -p -- "${WORK_DIR}" && cd -- "${WORK_DIR}" || {
    echo "Error: cannot create or enter ${WORK_DIR}" >&2
    exit 1
}

# Ensure snapshot mode (ORBIT_OPTIMIZED_CONF unset)
unset ORBIT_OPTIMIZED_CONF

# Thread counts: powers of two from MIN_THREADS up to the number of available CPUs, plus the CPU count
# itself when it is not a power-of-two step. Override the start with e.g. MIN_THREADS=8 ./scan.sh ...
MIN_THREADS="${MIN_THREADS:-16}"
MAX_THREADS="$(nproc)"
THREADS=()
for ((i=MIN_THREADS; i<=MAX_THREADS; i*=2)); do
    THREADS+=("$i")
done
if [[ ${#THREADS[@]} -eq 0 || "${THREADS[-1]}" != "${MAX_THREADS}" ]]; then
    THREADS+=("${MAX_THREADS}")
fi

# OpenMP scheduling policies
SCHEDULES=("static" "dynamic")

# Chunk sizes to test, spaced ~4x apart from fine-grained to coarse (override with e.g. CHUNK_LIST="1 8 64")
read -r -a CHUNKS <<< "${CHUNK_LIST:-1 16 64 256 1024}"

# CPU frequencies, all in kHz as reported by sysfs (sweep from the max frequency down to 1GHz).
# The max may be a turbo marker such as 2601000 (nominal + 1000kHz), so it is used verbatim.
CPUFREQ_DIR="/sys/devices/system/cpu/cpu0/cpufreq"
START_FREQ_KHZ=1000000
STEP_KHZ=200000
FREQUENCIES=()

if [[ -r "${CPUFREQ_DIR}/cpuinfo_min_freq" && -r "${CPUFREQ_DIR}/cpuinfo_max_freq" ]]; then
    MIN_FREQ_KHZ=$(<"${CPUFREQ_DIR}/cpuinfo_min_freq")
    MAX_FREQ_KHZ=$(<"${CPUFREQ_DIR}/cpuinfo_max_freq")

    # Round start frequency down to nearest clean STEP_KHZ boundary (e.g. 1210811 -> 1200000)
    # If rounded is below START_FREQ_KHZ (1GHz), clamp to START_FREQ_KHZ
    rounded_start=$(( (MIN_FREQ_KHZ / STEP_KHZ) * STEP_KHZ ))
    if (( rounded_start < START_FREQ_KHZ )); then
        rounded_start=${START_FREQ_KHZ}
    fi

    # Round max frequency down to nearest clean STEP_KHZ boundary (e.g. 4410811 -> 4400000)
    rounded_max=$(( (MAX_FREQ_KHZ / STEP_KHZ) * STEP_KHZ ))

    # Frequencies run from the highest to the lowest. The exact hardware max frequency (possibly a
    # turbo marker) comes first when it differs from the highest stepped frequency.
    if (( MAX_FREQ_KHZ != rounded_max )); then
        FREQUENCIES+=("${MAX_FREQ_KHZ}KHz")
    fi

    for ((f=rounded_max; f>=rounded_start; f-=STEP_KHZ)); do
        FREQUENCIES+=("${f}KHz")
    done
else
    echo "[WARNING] ${CPUFREQ_DIR} not available, using default CPU frequency" >&2
fi

# Handle empty frequencies array gracefully
if [[ ${#FREQUENCIES[@]} -eq 0 ]]; then
    FREQUENCIES=("")
fi

echo "============================================================"
echo "ORBIT Snapshot Parameter Sweep"
echo "Target program : $*"
echo "Working dir    : ${WORK_DIR}"
echo "ORBIT library  : ${ORBIT_LIB}"
echo "Threads        : ${THREADS[*]}"
echo "Schedules      : ${SCHEDULES[*]}"
echo "Chunks         : ${CHUNKS[*]}"
echo "Frequencies    : ${FREQUENCIES[*]:-default}"
# With a single thread the schedule and chunk have no effect, so it runs only once per frequency
# (with the first schedule and chunk).
TOTAL_RUNS=0
for threads in "${THREADS[@]}"; do
    if [[ "${threads}" == "1" ]]; then
        TOTAL_RUNS=$(( TOTAL_RUNS + 1 ))
    else
        TOTAL_RUNS=$(( TOTAL_RUNS + ${#SCHEDULES[@]} * ${#CHUNKS[@]} ))
    fi
done
TOTAL_RUNS=$(( TOTAL_RUNS * ${#FREQUENCIES[@]} ))

echo "Total iterations: ${TOTAL_RUNS}"
echo "============================================================"

for freq in "${FREQUENCIES[@]}"; do
    for sched in "${SCHEDULES[@]}"; do
        for chunk in "${CHUNKS[@]}"; do
            for threads in "${THREADS[@]}"; do
                if [[ "${threads}" == "1" && ( "${sched}" != "${SCHEDULES[0]}" || "${chunk}" != "${CHUNKS[0]}" ) ]]; then
                    continue
                fi
                echo ""
                echo ">>> Running: FREQ=${freq:-default} | SCHEDULE=${sched},${chunk} | THREADS=${threads}"
                
                cmd_status=0
                if [[ -n "${freq}" ]]; then
                    OMP_NUM_THREADS="${threads}" \
                    OMP_SCHEDULE="${sched},${chunk}" \
                    OMP_DYNAMIC=false \
                    ORBIT_CPU_FREQ="${freq}" \
                    LD_PRELOAD="${ORBIT_LIB}" \
                    "$@" || cmd_status=$?
                else
                    OMP_NUM_THREADS="${threads}" \
                    OMP_SCHEDULE="${sched},${chunk}" \
                    OMP_DYNAMIC=false \
                    LD_PRELOAD="${ORBIT_LIB}" \
                    "$@" || cmd_status=$?
                fi

                if [[ ${cmd_status} -ne 0 ]]; then
                    echo "[WARNING] Run exited with status ${cmd_status}" >&2
                fi
            done
        done
    done
done

echo ""
echo "============================================================"
echo "ORBIT Parameter Sweep Complete!"
echo "Visualize energy and PMU time-series results with:"
echo "  python3 ${SCRIPT_DIR}/tools/orbit-viz.py ${WORK_DIR}/<run_folder>"
echo "  python3 ${SCRIPT_DIR}/tools/orbit-viz.py --terminal ${WORK_DIR}/<run_folder>"
echo "============================================================"

