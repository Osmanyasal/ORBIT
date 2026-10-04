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
On start ORBIT looks for a configuration file (`ORBIT_OPTIMIZED_CONF`, then
`ORBIT_CONFIG`, default `optimized.conf`) and picks its mode:

- **Optimize** (file exists): the per-region `threads`, `sched`, `chunk` and
  `frequency` it contains are applied to the matching regions at runtime.
- **Snapshot** (no file): regions run with the settings given by the
  environment and are profiled; each region is appended to
  `snapshot.conf` in the OPTKIT execution folder. Run the application several
  times, varying `OMP_NUM_THREADS`, `OMP_SCHEDULE` (schedule and chunk) and
  `ORBIT_CPU_FREQ` (MHz, or with a unit such as `2.4GHz`), to compare settings.

```bash
# snapshot analysis (no optimized.conf present)
OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,4 ORBIT_CPU_FREQ=2400 \
LD_PRELOAD="$PWD/bin/Release/liborbit.so" ./your_application

# apply the chosen settings
ORBIT_OPTIMIZED_CONF=optimized.conf LD_PRELOAD="$PWD/bin/Release/liborbit.so" ./your_application
```

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
