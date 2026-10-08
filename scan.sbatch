#!/usr/bin/env bash
#SBATCH --job-name=orbit-scan
#SBATCH --partition=p12-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment=use:cpu-scaling=true
#SBATCH --mail-user=osman.yasal@vsb.cz
#SBATCH --mail-type=ALL
#SBATCH --output=orbit-scan-%j.out
#SBATCH --error=orbit-scan-%j.err

# Usage: edit PROGRAM below, then submit with
#   sbatch /path/to/ORBIT/scan.sbatch
#
# The sweep runs in <submit directory>/<executable name>. Use absolute paths for any file arguments.
# Overridable environment variables: ORBIT_ROOT, ORBIT_LIB, MIN_THREADS, CHUNK_LIST.

# Program to sweep: the executable followed by its arguments (a relative path resolves against the submit directory).
PROGRAM=(/home/yas0014/orbit/ORBIT/bin/Release/orbit_test)

set -u -o pipefail

ORBIT_ROOT="${ORBIT_ROOT:-/home/yas0014/orbit/ORBIT}"
SCAN="${ORBIT_ROOT}/scan.sh"

if [[ ! -x "${SCAN}" ]]; then
    echo "Error: ${SCAN} not found or not executable (set ORBIT_ROOT)" >&2
    exit 1
fi

# sbatch starts the job in the submit directory, so a relative executable resolves against it.
cd -- "${SLURM_SUBMIT_DIR:-${PWD}}" || exit 1

echo "Job ${SLURM_JOB_ID:-n/a} on $(hostname), $(nproc) CPUs, started $(date -Is)"
echo "Submit dir: ${PWD}"

"${SCAN}" "${PROGRAM[@]}"
status=$?

echo "scan.sh finished with status ${status} at $(date -Is)"
exit "${status}"
