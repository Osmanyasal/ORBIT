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

# Ensure snapshot mode (ORBIT_OPTIMIZED_CONF unset)
unset ORBIT_OPTIMIZED_CONF

# Thread counts to test (modify or add values as needed)
THREADS=(1 2 4 8)

# OpenMP scheduling policies
SCHEDULES=("static" "dynamic" "guided")

# Chunk sizes to test
CHUNKS=(1 4 8 16 32 64 128 256 512 1024)

# CPU frequencies (in MHz, e.g. 2400, or with unit like 2.4GHz)
# Add your target frequencies to this list:
FREQUENCIES=(
    2400    ## 2.4GHz
    2000    ## 2.0GHz
)

# Handle empty frequencies array gracefully
if [[ ${#FREQUENCIES[@]} -eq 0 ]]; then
    FREQUENCIES=("")
fi

echo "============================================================"
echo "ORBIT Snapshot Parameter Sweep"
echo "Target program : $*"
echo "ORBIT library  : ${ORBIT_LIB}"
echo "Threads        : ${THREADS[*]}"
echo "Schedules      : ${SCHEDULES[*]}"
echo "Chunks         : ${CHUNKS[*]}"
echo "Frequencies    : ${FREQUENCIES[*]:-default}"
echo "============================================================"

for freq in "${FREQUENCIES[@]}"; do
    for threads in "${THREADS[@]}"; do
        for sched in "${SCHEDULES[@]}"; do
            for chunk in "${CHUNKS[@]}"; do
                echo ""
                echo ">>> Running: THREADS=${threads} | SCHEDULE=${sched},${chunk} | FREQ=${freq:-default}"
                
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
