# ORBIT

<img width="1408" height="768" alt="Gemini_Generated_Image_rmsyt3rmsyt3rmsy" src="https://github.com/user-attachments/assets/55260ad1-ca70-4857-9ea9-78fdc92b3240" />

OpenMP Runtime Backend for Intelligent Tuning

## Download and Install 🚀

```bash
git clone https://github.com/Osmanyasal/ORBIT.git
git submodule update --init --recursive

pushd lib/OPTKIT/lib/premake5
make -f Bootstrap.mak linux
alias premake5="$PWD/bin/release/premake5"
popd

premake5 gmake
make -C build config=release orbit orbit_test
```

## OpenMP Backends

The default backend is GCC/libgomp and does not require libffi. Regenerate
makefiles after changing backend options:

```bash
premake5 --openmp-backend=gcc gmake
make -C build orbit
```

For Intel/libiomp5 or Clang/libomp, select `--openmp-backend=intel` or
`--openmp-backend=clang`. Both use the shared KMP ABI hooks and require the
libffi development package (headers and linker library), not just its runtime
library. `--openmp-backend=all` exports both ABI families and also requires
libffi. Select the backend for the application being preloaded, not necessarily
the compiler used to build ORBIT.

If libffi is installed in a custom prefix:

```bash
premake5 --openmp-backend=clang --libffi-root=/path/to/libffi gmake
make -C build orbit
```

Some distributions put libffi headers in a versioned directory; ensure that
directory is on the compiler's include path. On RPM-based systems, the system
development package is commonly named `libffi-devel`; on Debian/Ubuntu it is
`libffi-dev`. Installing system packages requires administrator access.

## Per-Region OpenMP Interception

Preload `liborbit.so` into an application using the selected OpenMP runtime.
On start ORBIT checks `ORBIT_OPTIMIZED_CONF` and picks its mode:

- **Optimize** (`ORBIT_OPTIMIZED_CONF` names an existing file): the per-region `threads`, `sched`, `chunk` and
  `frequency` it contains are applied to the matching regions at runtime.
- **Snapshot** (variable unset or file missing): regions run with the settings given by the
  environment and are profiled; each region is appended to
  `snapshot.conf` in the OPTKIT execution folder. Run the application several
  times, varying `OMP_NUM_THREADS`, `OMP_SCHEDULE` (schedule and chunk) and
  `ORBIT_CPU_FREQ` (MHz, or with a unit such as `2.4GHz`), to compare settings.

```bash
# snapshot analysis (ORBIT_OPTIMIZED_CONF unset)
OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,4 ORBIT_CPU_FREQ=2400 \
LD_PRELOAD="$PWD/bin/Release/liborbit.so" ./your_application

# apply the chosen settings
ORBIT_OPTIMIZED_CONF=optimized.conf LD_PRELOAD="$PWD/bin/Release/liborbit.so" ./your_application
```

### Snapshot Profiling Considerations

In Snapshot mode, the interceptor enables both screenshot collection (`is_screenshot = true`) and periodic sampling (`is_sampling = true` with 1-second intervals) to capture time-series datasets. This mode is designed for parallel regions running for at least 1 second:

| Aspect | Parallel Region < 1 second | Parallel Region $\ge$ 1 second |
| :--- | :--- | :--- |
| **Time-Series Samples** | Only 1–2 samples; no time-series progression | Produces sequential 1-second interval samples |
| **Exit Latency** | Destructor blocks up to ~1s in thread join | Negligible relative overhead (joins fractional sleep) |
| **Dataset Usability (train)** | Insufficient sequence length for GRU model | Valid time-series dataset |
| **Memory Footprint** | Low (only 1–2 samples) | Grows linearly with execution time ($O(N)$ in RAM) |

Edit `region_begin` and `region_end` directly in `src/orbit.cc`. They run on
the thread initiating the region, immediately before the real runtime call and after its team finishes. For legacy GCC split
regions, begin runs at `GOMP_parallel_start` and end after `GOMP_parallel_end`.
The outlined function and caller address identify the region within the process.

```cpp
static void region_begin(Region& region) {
    region.threads = 4;
}

static void region_end(const Region& region) {
}
```

`Region` carries `function`, `caller`, `entry` (the intercepted API name),
`threads`, and `chunk`. Changing `threads` changes the forwarded thread request;
zero lets libgomp use its normal default. Explicit one-thread requests,
including serialized `if(0)` regions, stay single-threaded. `chunk` is forwarded
only by combined parallel-loop APIs with an explicit chunk argument; runtime
loops have no such argument. Rebuild after changing the hooks. Synchronize any
shared state you add, since independent/nested regions may run concurrently.

Only parallel-region boundaries are intercepted: ordinary parallel calls,
legacy parallel start/end, modern combined parallel loops (including
nonmonotonic/runtime variants), and modern parallel sections for GCC. The
Intel/Clang headers intercept `__kmpc_fork_call` and serialized parallel
begin/end. Their fork calls are variadic; libffi forwards the captured pointers
without an arbitrary argument limit. Serialized KMP boundaries do not expose
the outlined function, so `Region::function` is null for those calls. KMP
`chunk` is not forwarded by fork calls; use the runtime schedule API when
appropriate. Standalone loop
start/next calls are not intercepted, and scheduling policies are not remapped.
For runtime-scheduled loops, you can set the OpenMP runtime schedule yourself
in the hook if needed. Compiler-inlined static scheduling cannot be changed
through preload. Statically linked runtimes, legacy combined start APIs,
teams-specific fork calls, taskloops, and offload are not covered. Intel/Clang
native-runtime testing has not been performed in the current environment.

## Tests

```bash
LD_PRELOAD="$PWD/bin/Release/liborbit.so" \
OMP_DYNAMIC=false ./bin/Release/orbit_test
```

## Energy & PMU Time-Series Visualization 📊

ORBIT includes a dedicated visualization tool (`tools/orbit-viz.py`) for analyzing energy consumption, power profiles, and PMU metric time-series changes across periodic execution intervals and parameter sweeps.

It has **zero external package dependencies** (runs out-of-the-box on standard Python 3.6+) and supports interactive HTML dashboards, terminal sparklines, static SVG/PNG vector exports, and live HTTP serving.

```bash
# 1. Generate an interactive offline HTML dashboard (default: orbit_report.html)
python3 tools/orbit-viz.py <run_directory>

# 2. View rich ANSI terminal summary tables and Unicode sparklines
python3 tools/orbit-viz.py --terminal <run_directory>

# 3. Start a local HTTP server to interactively explore the dashboard
python3 tools/orbit-viz.py --serve 8080 <run_directory>

# 4. Export static SVG and PNG figures (dual-axis power/IPC and cross-region comparison)
python3 tools/orbit-viz.py --export-plots plots/ <run_directory>

# 5. Output processed metrics and step changes as JSON
python3 tools/orbit-viz.py --json <run_directory> > report.json
```

### Visualized Metrics & Changes

- **Continuous Program Sequence**: Chronological execution timeline chaining all parallel regions based on file creation time ordering, with region transition boundaries, program sequence ribbon, and full-run energy/power/PMU evolution.
- **Energy & Power Profile**: Total package energy (Joules), watt-hours, socket breakdown, average power (Watts), and Energy-Delay Product (EDP $J \cdot s$).
- **PMU Performance Progression**: Instructions Per Cycle (IPC), core frequency (GHz), instruction throughput (GIPS), L2 cache hit ratio (%), L3 MPKI (misses per 1k instructions), and branch misprediction ratio.
- **Dynamic Change ($\Delta$)**: Step-by-step performance shifts ($\Delta \text{metric}$), rates of change ($dM/dt$), percentage change ($\% \Delta$), and phase transition tracking.
- **Cross-Region & Sweep Comparisons**: Side-by-side Pareto efficiency comparisons across parallel regions and thread/schedule/chunk configurations.

