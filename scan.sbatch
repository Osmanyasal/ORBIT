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
# The thread count is NOT an argument: scan.sh sets OMP_NUM_THREADS for every run, and the benchmarks read it from
# the environment. Where a rodinia benchmark still requires a positional thread-count argument (bfs, hotspot,
# leukocyte, sc_omp, srad_v1, srad_v2) it is a placeholder (1) that OMP_NUM_THREADS overrides.
# PROGRAM=(/home/yas0014/orbit/ORBIT/bin/Release/orbit_test)
# PROGRAM=(/home/yas0014/orbit/benchmarks/NPB3.4.4/NPB3.4-OMP/bin/bt.C.x)
# PROGRAM=(/home/yas0014/orbit/benchmarks/NPB3.4.4/NPB3.4-OMP/bin/cg.C.x)
# PROGRAM=(/home/yas0014/orbit/benchmarks/NPB3.4.4/NPB3.4-OMP/bin/ep.D.x)
# PROGRAM=(/home/yas0014/orbit/benchmarks/NPB3.4.4/NPB3.4-OMP/bin/lu.C.x)
# PROGRAM=(/home/yas0014/orbit/benchmarks/NPB3.4.4/NPB3.4-OMP/bin/mg.D.x)
# PROGRAM=(/home/yas0014/orbit/benchmarks/NPB3.4.4/NPB3.4-OMP/bin/sp.C.x)
# PROGRAM=(/home/yas0014/orbit/benchmarks/CoMD/bin/CoMD-openmp -x 70 -y 70 -z 70 -N 100 -n 10)
# PROGRAM=(/home/yas0014/orbit/benchmarks/gapbs/bfs -f /home/yas0014/orbit/benchmarks/gapbs/benchmark/graphs/kron23.sg -n 16)
# PROGRAM=(/home/yas0014/orbit/benchmarks/gapbs/pr -f /home/yas0014/orbit/benchmarks/gapbs/benchmark/graphs/kron23.sg -n 8)
# PROGRAM=(/home/yas0014/orbit/benchmarks/gapbs/cc -f /home/yas0014/orbit/benchmarks/gapbs/benchmark/graphs/kron23.sg -n 16)
# PROGRAM=(/home/yas0014/orbit/benchmarks/LULESH/lulesh2.0 -s 100 -i 10)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/bfs 1 /home/yas0014/orbit/benchmarks/rodinia/data/bfs/graph16.txt)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/euler3d_cpu /home/yas0014/orbit/benchmarks/rodinia/data/cfd/fvcorr.domn.097K)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/euler3d_cpu_double /home/yas0014/orbit/benchmarks/rodinia/data/cfd/fvcorr.domn.097K)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/pre_euler3d_cpu /home/yas0014/orbit/benchmarks/rodinia/data/cfd/fvcorr.domn.097K)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/pre_euler3d_cpu_double /home/yas0014/orbit/benchmarks/rodinia/data/cfd/fvcorr.domn.097K)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/heartwall /home/yas0014/orbit/benchmarks/rodinia/data/heartwall/test.avi 104)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/hotspot 1024 1024 100000 1 /home/yas0014/orbit/benchmarks/rodinia/data/hotspot/temp_1024 /home/yas0014/orbit/benchmarks/rodinia/data/hotspot/power_1024 /dev/null)

# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/kmeans -i /home/yas0014/orbit/benchmarks/rodinia/data/kmeans/kdd_cup -k 100)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/lavaMD -boxes1d 64)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/lud_omp -s 24000)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/pathfinder 4000000 200)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/sc_omp 10 20 256 262144 262144 1000 none /dev/null 1)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/srad_v1 600 0.5 4096 4096 1)
# PROGRAM=(/home/yas0014/orbit/benchmarks/rodinia/bin/linux/omp/srad_v2 4096 4096 0 127 0 127 1 0.5 1000)

set -u -o pipefail

ORBIT_ROOT="${ORBIT_ROOT:-/home/yas0014/orbit/ORBIT}"
SCAN="${ORBIT_ROOT}/scan.sh"

if [[ ! -x "${SCAN}" ]]; then
    echo "Error: ${SCAN} not found or not executable (set ORBIT_ROOT)" >&2
    exit 1
fi

# The sweep runs in the current directory: sbatch starts the job in the submit directory, and when the
# script is run by hand (e.g. inside salloc) it uses the directory you are in.
cd ${PWD}/..
echo "Job ${SLURM_JOB_ID:-n/a} on $(hostname), $(nproc) CPUs, started $(date -Is)"
echo "Working directory: ${PWD}"

"${SCAN}" "${PROGRAM[@]}"
status=$?

echo "scan.sh finished with status ${status} at $(date -Is)"
exit "${status}"
