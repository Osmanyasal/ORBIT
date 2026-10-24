# ORBIT

<img width="100%" alt="ORBIT logo: interwoven orbital threads around a parallel-computing core beside the ORBIT wordmark on a white background" src="assets/orbit-logo.svg" />

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

## Quick Start: Tune an Application 🎯

Two scripts in `tools/` automate the whole loop: analyse a program once, predict the best settings
for every OpenMP region, and run the program with those settings applied. Run them on a node where
`liborbit.so` can load (it needs the GPU libraries it was built against).

Requirements: a built `bin/Release/liborbit.so`, a Python environment with `torch`, `pandas` and `numpy`
(`tools/venv-orbit`) and trained models in `tools/models_best/` (`model_fastest.pt`, `model_best_edp.pt`;
train them with `tools/run_predictor_sweep.sh`, see [Configuration Predictor](#configuration-predictor-mlp-)).

```bash
# 0. One-time: create the virtual environment (CPU torch, numpy, pandas) with Python >= 3.9.
#    orbit-analyze.sh runs this automatically when no working Python is found. The environment belongs to
#    the node's Python, so repeat it on a node where the existing tools/venv-orbit does not work.
tools/setup-venv.sh            # --force recreates it; PYTHON=/path/to/python3.x picks the interpreter

# 1. Analyse: one Snapshot run + prediction -> orbit_<app>/best_per_region.json
#    (holds the `fastest` and `best_edp` settings of every region)
tools/orbit-analyze.sh -- ./your_application args...

# 2. Optimize: run with the settings applied
tools/orbit-optimize.sh -p edp -- ./your_application args...          # best EDP (default)
tools/orbit-optimize.sh -p fastest -- ./your_application args...      # fastest
tools/orbit-optimize.sh -p both --baseline -- ./your_application args...  # both, plus an untuned run
```

`orbit-analyze.sh` runs the program once in Snapshot mode (`ORBIT_OPTIMIZED_CONF` unset; the run uses
`OMP_NUM_THREADS`, `OMP_SCHEDULE` and `ORBIT_CPU_FREQ` from the environment), then calls
`orbit-predictor.py generate` on the resulting execution folder.

| Option | Description |
|--------|-------------|
| `-o, --out DIR` | Output directory (default: `./orbit_<application name>`) |
| `-m, --model-dir DIR` | Trained models (default: `tools/models_best`, or `$ORBIT_MODEL_DIR`) |
| `--max-threads N`, `--freq-range MIN_KHZ MAX_KHZ`, `--freq-step KHZ` | Describe the target machine when cpufreq is not readable (defaults: `nproc`, cpufreq range, 200 MHz) |
| `--skip-first N`, `--stat mean\|median`, `--time-source energy\|pmu` | How the calls of a region are compressed |
| `--keep-run DIR` | Analyse an existing execution folder instead of running the program |

`orbit-optimize.sh` measures every optimized and optional baseline run with
`optkit-cli stat --cpu-energy -- <program>`. Application output and OPTKIT's CPU energy
report are printed for each run, followed by a wall-clock summary. ORBIT is preloaded
only into the optimized application, not the profiler.

Build the CLI against the built OPTKIT library before optimizing:

```bash
make -C lib/OPTKIT/tools/optkit-cli
```

The default executable is `lib/OPTKIT/tools/optkit-cli/optkit-cli`; set `OPTKIT_CLI`
to another executable path or a command in `PATH` if needed. Energy measurement
requires access to the CPU energy counters or the energy backend configured in OPTKIT.

| Option | Description |
|--------|-------------|
| `-c, --conf FILE` | Settings file (default: `./orbit_<application name>/best_per_region.json`) |
| `-p, --policy` | `edp` (default), `fastest` or `both` (one run each) |
| `-t, --threshold T` | `ORBIT_TIME_THRESHOLD`, e.g. `10ms`; `0` tunes every region |
| `--baseline` | Also run the program once without ORBIT |

It stops with an error if the settings file is missing, because ORBIT would otherwise silently fall back to
Snapshot mode. Both scripts accept `ORBIT_LIB` to use another `liborbit.so`, and `--help` lists every option.
The wall-clock summary includes input loading and profiler overhead, so it is only a rough timing
check. Use OPTKIT's measured CPU energy and duration for whole-application energy/EDP comparisons;
Snapshot measurements remain useful for per-region comparisons. Application and profiler failures
stop the script rather than producing a successful timing summary.

## Per-Region OpenMP Interception

Preload `liborbit.so` into an application using the selected OpenMP runtime.
On start ORBIT checks `ORBIT_OPTIMIZED_CONF` and picks its mode:

- **Optimize** (`ORBIT_OPTIMIZED_CONF` names an existing file, typically the `best_per_region.json` written by
  `tools/best_settings.py`): for every matching region ORBIT loads the evaluated `fastest` and `best_edp`
  candidates and applies one of them (`threads`, `sched`, `chunk`, `frequency`) at runtime, as selected by
  `ORBIT_OPTIMIZE_POLICY` (see below). Regions whose total execution time (either best runtime or best edp) is not above the threshold
  (default 100 ms) are skipped.
- **Snapshot** (variable unset or file missing): regions run with the settings given by the
  environment and are profiled; each region is appended to
  `snapshot.conf` in the OPTKIT execution folder the first time it runs with a given set of settings
  (`threads`, `sched`, `chunk`, `frequency`), so a region entered many times with the same settings has a
  single entry. Run the application several
  times, varying `OMP_NUM_THREADS`, `OMP_SCHEDULE` (schedule and chunk) and
  `ORBIT_CPU_FREQ` (MHz, or with a unit such as `2.4GHz`), to compare settings.
  A region entered repeatedly (e.g. inside a time-step loop) is profiled on every entry; each entry is
  appended as a new element of the region's single `*_cpu_pmu.json` / `*_cpu_energy.json` array, in
  execution order.

```bash
# snapshot analysis (ORBIT_OPTIMIZED_CONF unset)
OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,4 ORBIT_CPU_FREQ=2400 \
LD_PRELOAD="$PWD/bin/Release/liborbit.so" ./your_application

# apply the chosen settings (lowest-EDP candidate by default)
ORBIT_OPTIMIZED_CONF=best_per_region.json LD_PRELOAD="$PWD/bin/Release/liborbit.so" ./your_application

# apply the fastest candidate instead, and also tune regions down to 10 ms
ORBIT_OPTIMIZED_CONF=best_per_region.json ORBIT_OPTIMIZE_POLICY=fastest ORBIT_TIME_THRESHOLD=10ms \
LD_PRELOAD="$PWD/bin/Release/liborbit.so" ./your_application
```

### Optimize Mode Environment Variables

| Variable | Description |
|----------|-------------|
| `ORBIT_OPTIMIZED_CONF` | Path to the configuration file (`best_per_region.json`). If set and the file exists, ORBIT runs in Optimize mode; otherwise in Snapshot mode. |
| `ORBIT_OPTIMIZE_POLICY` | Which evaluated candidate is applied to each region: `edp` (default; `best_edp` is accepted as an alias) applies the lowest-EDP settings, `fastest` applies the lowest-time settings. If the preferred candidate is missing for a region, the other one is used. |
| `ORBIT_TIME_THRESHOLD` | Minimum total execution time of a region (`total_time_s`) for it to be tuned. Accepts `50ms`, `0.05s` or a plain number (values >= 10 are milliseconds, smaller values seconds). Default `50ms`; `0` tunes every region. |

The file is read into `Region::fastest` and `Region::best_edp` (with `calls` and `configs_evaluated`);
the chosen candidate is then copied into `Region::current` when the region starts, before `region_begin` runs.
A configuration file holding a single flat setting per region (the `snapshot.conf` format) is also accepted;
that setting is used as both candidates.

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

## Preparing Benchmarks for the Sweep

`scan.sh` varies `OMP_SCHEDULE` (schedule and chunk), which ORBIT applies with `omp_set_schedule()`. That only affects loops written with `schedule(runtime)`; loops with a fixed schedule or none ignore it, so the sweep would show no effect of `sched`/`chunk`. The NPB sources already use `schedule(runtime)`. For other applications run `tools/patch_schedule_runtime.py` on their source before building:

```bash
python3 tools/patch_schedule_runtime.py <src_dir> [<src_dir> ...] [--dry-run] [--backup] [--report FILE]
```

It adds `schedule(runtime)` to worksharing loops (`for`, `parallel for`, Fortran `do`, `parallel do`, with or without `simd`) that have no schedule clause, and replaces `schedule(static|dynamic|guided|auto[, n])` by `schedule(runtime)`. C, C++ and free-form Fortran (`.f90`/`.f95`/`.f03`) are handled, including multi-line directives. Offload (`target`, `teams`, `distribute`), `simd`, `task` and `sections` constructs are left alone, and a second run changes nothing.

| Option | Description |
|--------|-------------|
| `--dry-run` | Report what would change, write nothing |
| `--backup` | Keep `<file>.orig` for every modified file (not needed inside a git repository: use `git diff` / `git checkout .`) |
| `--report FILE` | Write one line per loop: `file:line: action \| original directive` (`added`, `replaced schedule(...)`, `already runtime` or `skipped: ...`) |
| `--exclude TEXT` | Skip files whose path contains `TEXT` (repeatable) |
| `--allow-nowait TEXT` | Also patch `nowait` loops in files whose path contains `TEXT` (repeatable; review them first) |

Loops with `nowait`, and the loops that follow one up to the next barrier or parallel region, are skipped and listed, unless they already use a dynamic, guided or auto schedule. With a static schedule an iteration runs on the same thread in consecutive loops, so a loop may depend on data written by a preceding `nowait` loop without a barrier; with dynamic scheduling that becomes a data race.

Not handled, to be fixed by hand: fixed-form Fortran (`.f`), loops hidden in macros (the schedule inside a `#define`), and `omp_set_num_threads()` calls that override `OMP_NUM_THREADS`. The script never builds anything. After building, confirm that `sched` and `chunk` change the region times (for example `OMP_SCHEDULE=static,1`, `dynamic,1` and `dynamic,1024`) before starting a sweep.

## Best-Settings Search & Heatmaps 🔥

Two companion tools process a parameter sweep (one sub-folder per execution, each with a `snapshot.conf` plus the per-region `*__cpu_energy.json` / `*__cpu_pmu.json` files) and find the best OpenMP settings per parallel region.

### `tools/best_settings.py`

Compresses all calls of a region inside one execution into a single value (mean or median), merges repeated runs with identical settings, and reports for every region the **fastest** and the **lowest-EDP** configuration (EDP = energy [J] × time [s]).

```bash
python3 tools/best_settings.py <sweep_dir> [-o out_dir] [--stat mean|median] \
                               [--skip-first N] [--time-source energy|pmu] [--top K] [-j JOBS]
```

| Option | Description |
|--------|-------------|
| `-o, --out` | Output directory (default: `best_settings_<sweep_dir>` next to the sweep folder) |
| `--stat` | How repeated calls of a region are compressed (default: `mean`) |
| `--skip-first N` | Ignore the first N calls of each region (warm-up) |
| `--time-source` | Take the duration from the `energy` (default) or `pmu` measurement |
| `--top K` | Show the top-K settings per region in the terminal summary (default: 3) |
| `-j, --jobs` | Parallel workers (default: all CPUs) |

Outputs in the output directory:

- `all_configs.csv`: one row per (region, threads, sched, chunk, frequency) with `calls`, per-call `time_s`, `energy_j`, `edp`, `power_w` and `ipc`.
- `best_per_region.json`: fastest and best-EDP settings per region, with both per-call values and **totals over all calls** (per-call value × `calls`; total EDP = total time × total energy), plus a `_total` entry for the whole program.

The terminal summary also prints estimated program-level totals (per-region best vs. best single uniform configuration).

### `tools/plot_heatmaps.py`

Reads `all_configs.csv` and writes two heatmaps per region: execution time (`<region>_time.<fmt>`) and **K-EDP** (`<region>_edp.<fmt>`), i.e. EDP divided by 1000, in kJ·s. The Y axis is the frequency, the X axis is sched/threads/chunk, and the colour scale is logarithmic (plain decimal tick labels). A star marks the fastest configuration and a diamond the lowest-K-EDP one. The legend next to each marker lists time, energy and K-EDP both **per call** and **total** (per call × number of calls).

```bash
python3 tools/plot_heatmaps.py <best_settings_dir> [-o out_dir] [--total] [--linear] [--format png|pdf|svg]
```

| Option | Description |
|--------|-------------|
| `-o, --out` | Output directory (default: `<best_settings_dir>/heatmaps`) |
| `--total` | Colour the heatmaps by totals over all calls instead of per-call values (legend always shows both) |
| `--linear` | Linear colour scale instead of logarithmic |
| `--format` | Output format: `png` (default), `pdf` or `svg` |

Requires `matplotlib` and `numpy`.

```bash
python3 tools/best_settings.py runs/bt.C.x
python3 tools/plot_heatmaps.py best_settings_bt.C.x
```

### Automated workflow (`tools/orbit-analyze.sh`, `tools/orbit-optimize.sh`)

See [Quick Start](#quick-start-tune-an-application-) at the top of this file.

## Configuration Predictor (MLP) 🧠

`tools/train_model.py` trains two MLPs (`fastest` and `best_edp`) on the `all_configs.csv` / `best_per_region.json` pairs written by `tools/best_settings.py`. From the PMU and energy metrics of a single snapshot execution (ipc, l2_hit_ratio, l3_mpki, branch_mispr_ratio), the models predict `threads`, `sched`, `chunk` and `frequency` per region, and `generate` writes them as a `best_per_region.json`-style file for Optimize mode. No time series is needed.

Requires `numpy`, `pandas` and `torch`.

```bash
cd tools

# 1. Train on every <data_dir>/**/ folder holding all_configs.csv + best_per_region.json
python3 train_model.py train --data-dir <data_dir> --out-dir models

# 2. Run the application once in Snapshot mode (ORBIT_OPTIMIZED_CONF unset), then predict from that execution folder
python3 train_model.py generate --model-dir models --snapshots <execution_folder> --output predicted_best_per_region.json

# 3. Apply the prediction
ORBIT_OPTIMIZED_CONF=predicted_best_per_region.json ORBIT_OPTIMIZE_POLICY=edp \
LD_PRELOAD="$PWD/../bin/Release/liborbit.so" ./your_application
```

| Command | Description |
|---------|-------------|
| `train` | Leave-one-application-out cross-validation, then trains and saves `model_fastest.pt`, `model_best_edp.pt` and `cv_metrics.json` in `--out-dir`. The per-application geometric-mean extra cost of the predictor, the default and the global fixed configuration is printed and saved as `cv_per_app_<target>.csv` in `--out-dir`. `--max-len` sets the maximum number of observations per sample, `--layers` the number of dense layers, `--cv-mode none` skips validation, `--targets` trains one model, `--features` selects the metrics. |
| `generate` | Reads snapshot execution folder(s) (`--snapshots`, a folder with `snapshot.conf` or a folder of them) and writes the configuration file. Each region needs both its `*__cpu_pmu.json` and `*__cpu_energy.json`. |
| `predict` | Prints or saves recommendations from any CSV in the `all_configs.csv` format (`--input`, `--output`). |

Model: a feed-forward MLP (`--layers` dense layers, default 3). The same dense stack encodes every observation (up to `--max-len` per region, in random order), the encodings are averaged, and linear heads predict the four settings. `--max-len 1` makes it a plain MLP on one observation, which is what one snapshot execution gives.

Because the encodings are averaged, the order of the observations carries no information. Checkpoints from the removed GRU model are rejected; retrain them.

Notes:

- `generate` writes `fastest` and `best_edp` for each region; `ORBIT_OPTIMIZE_POLICY` picks one. The region-level `total_time_s` (right after `calls`) is `calls` × the best per-call time measured in the snapshot (used for `ORBIT_TIME_THRESHOLD`), not a prediction. ORBIT also still accepts a `total_time_s` inside `fastest` / `best_edp`, as written by `best_settings.py`.
- **Portable settings.** `threads`, `chunk` and `frequency` are predicted as positions on a normalised axis (threads as a fraction of the machine's cores, chunk on a log2 axis up to the largest trained chunk, frequency as a fraction of the machine's frequency range), over fixed bins (`--thread-bins`, `--freq-bins`). The most likely bin (`--readout argmax`, default; `median`/`mean` interpolate between bins, which did not help in cross-validation) is converted with the **target machine's** core count and frequency range and snapped to the nearest valid setting. Because the bins are finer than the swept steps, the model can pick settings that were not in the training sweep (e.g. 2.3 GHz when only 2.2 and 2.4 GHz were measured, or 24 threads) and can be used on a machine with a different core count or frequency range. `sched` stays a plain classification over the schedules seen in training. Checkpoints trained before this change are rejected; retrain them.
- The target machine used by `generate` / `predict` defaults to what `scan.sh` sweeps on the current machine (`nproc` threads, the cpufreq range in 200 MHz steps). When predicting for another machine, or where cpufreq is unavailable, give it with `--max-threads`, `--freq-range MIN_KHZ MAX_KHZ` and `--freq-step`.
- The default model input contains only machine-independent metrics (ipc, l2_hit_ratio, l3_mpki, branch_mispr_ratio). Absolute metrics such as `time_s` and `power_w` can be added with `--features`, but they depend on the hardware.
- Each training application is normalised with the settings its own sweep covered, so applications swept on different machines can share one `--data-dir`.
- Validation on unseen applications (leave-one-application-out, one observation per region) compares the predicted configuration with the best configuration of each region and with the untuned default (max threads, static, max frequency, chunk `--default-chunk`, 256 by default; the sweep has no unchunked `static`). The prediction is clearly closer to the optimum than the default; see the printed report and `models/cv_metrics.json`. More training applications are the most likely way to improve it.
