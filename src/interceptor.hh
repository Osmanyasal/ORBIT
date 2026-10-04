#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/stat.h>
#include <exception>
#include <memory>
#include <mutex>
#include <omp.h>
#include <string>
#include <unordered_map>
#include "optkit.hh"
#include "region.hh"
#include "utils.hh"


namespace orbit
{
    static void region_begin(Region &region);
    static void region_end(const Region &region);

    namespace detail
    {
        // Snapshot: no configuration file was found, so regions run with the settings supplied by the
        // environment (OMP_NUM_THREADS, OMP_SCHEDULE, ORBIT_CPU_FREQ) and are profiled and recorded.
        // Optimize: a configuration file exists and its per-region settings are applied at runtime.
        enum class Mode
        {
            Snapshot,
            Optimize
        };

        struct Runtime
        {
            Mode mode = Mode::Snapshot;
            std::unordered_map<std::string, Region> configs;
            std::int64_t frequency = 0;
        };

        // Process-wide OPTKIT instance, created on first use and destroyed at exit.
        // It initialises the logger and PMU query state the profilers rely on.
        inline std::unique_ptr<optkit::OPTKIT> &optkit_instance()
        {
            static std::unique_ptr<optkit::OPTKIT> instance;
            return instance;
        }

        inline Runtime load_runtime()
        {
            Runtime runtime;
            runtime.frequency = utils::requested_cpu_frequency_khz();

            const char *config_path = utils::optimized_config_path();
            struct stat config_status;
            if (config_path && 
                stat(config_path, &config_status) == 0 && 
                S_ISREG(config_status.st_mode))
            {
                runtime.mode = Mode::Optimize;
                runtime.configs = read_region_configs(config_path);
                std::fprintf(stderr, "ORBIT: optimize mode, %zu region(s) read from %s\n",
                             runtime.configs.size(), config_path);
            }
            else
            {
                runtime.mode = Mode::Snapshot;
                std::fprintf(stderr, "ORBIT: %s, snapshot mode active\n",
                             config_path ? "configuration file not found" : "ORBIT_OPTIMIZED_CONF not set");
            }

            const bool is_snapshot = runtime.mode == Mode::Snapshot;
            // Optimize mode changes the frequency per region, so the original must always be restored.
            const bool restore_freq = !is_snapshot || runtime.frequency > 0;
            optkit_instance().reset(new optkit::OPTKIT{optkit::OPTKIT_CONFIG{is_snapshot, "", restore_freq}});

            if (runtime.frequency > 0)
            {
                try
                {
                    for (int socket = 0; socket < OPTKIT_ENV_CPU_NUM_SOCKETS; ++socket)
                    {
                        optkit::frequency::cpu::Frequency::set_core_frequency(runtime.frequency, socket);
                    }
                    std::fprintf(stderr, "ORBIT: CPU frequency set to %lld kHz\n",
                                 static_cast<long long>(runtime.frequency));
                }
                catch (const std::exception &error)
                {
                    OPTKIT_ERROR("ORBIT: cannot set CPU frequency: {}", error.what());
                }
            }
            return runtime;
        }

        // Selects the mode, reads the configuration and initialises OPTKIT once, on first use.
        inline const Runtime &ensure_runtime()
        {
            static const Runtime runtime = load_runtime();
            return runtime;
        }
    }

    class RegionScope;
    inline thread_local RegionScope *active_region = nullptr;

    // Brackets one intercepted parallel region: region_begin on construction, region_end on destruction.
    // Scopes nest per thread through active_region; `split` marks heap-allocated scopes whose end is
    // signalled by a separate runtime call (legacy GOMP_parallel_start/end, serialized KMP regions).
    class RegionScope
    {
    public:
        Region region;
        bool split;

        RegionScope(void (*function)(), void *caller, const char *entry,
                    unsigned threads, long chunk = 0, bool legacy = false)
            : region{function, caller, entry,
                     threads ? static_cast<int>(threads) : omp_get_max_threads(),
                     chunk ? chunk : utils::current_chunk(), utils::current_sched(), utils::describe_caller(caller)},
              split(legacy),
              parent(active_region)
        {
            const detail::Runtime &runtime = detail::ensure_runtime();
            if (runtime.mode == detail::Mode::Optimize)
            {
                // if there is a configuration for this region, apply it
                const auto it = runtime.configs.find(region.name);
                if (it != runtime.configs.end())
                {
                    region.threads = it->second.threads;
                    region.chunk = it->second.chunk;
                    region.sched = it->second.sched;
                    region.frequency = it->second.frequency;
                }
            }
            else if (runtime.mode == detail::Mode::Snapshot)
            {
                region.frequency = runtime.frequency;   // set the region frequency to the runtime frequency in snapshot mode
                optkit::pmu::cpu::perf::PerfProfilerConfig perf_config{region.name.c_str(), true /*is_sampling*/};
                perf_config.is_screenshot = true;
                cpu_event_profiler.reset(new optkit::pmu::cpu::perf::BlockProfiler(perf_config, optkit::metrics::performance::cpu_metrics::ai()));
                cpu_energy_profiler.reset(new optkit::energy::rapl::Profiler(
                    {region.name.c_str(), "cpu_energy", true, true, optkit::Query::create_folder, !optkit::Query::create_folder},
                    optkit::metrics::energy::cpu_metrics::all_metrics()));
            }
            region_begin(region);
            active_region = this;
        }

        // Any other outlined-function signature (GOMP, KMP microtask) is erased to void (*)().
        template <typename Function>
        RegionScope(Function function, void *caller, const char *entry,
                    unsigned threads, long chunk = 0, bool legacy = false)
            : RegionScope(reinterpret_cast<void (*)()>(function), caller, entry, threads, chunk, legacy)
        {
        }

        ~RegionScope()
        {
            cpu_event_profiler.reset();
            cpu_energy_profiler.reset();
            active_region = parent;
            region_end(region);
            if (detail::ensure_runtime().mode == detail::Mode::Snapshot)
            {
                region.append_to_file(optkit::utils::EXECUTION_FOLDER_NAME + "/snapshot.conf");
            }
        }

        RegionScope(const RegionScope &) = delete;
        RegionScope &operator=(const RegionScope &) = delete;

    private:
        RegionScope *parent;
        std::unique_ptr<optkit::pmu::cpu::perf::BlockProfiler> cpu_event_profiler;
        std::unique_ptr<optkit::energy::rapl::Profiler> cpu_energy_profiler;
    };

    template <typename Function>
    inline Function resolve(const char *name)
    {
        void *address = dlsym(RTLD_NEXT, name);
        if (!address)
        {
            OPTKIT_ERROR("ORBIT: cannot resolve OpenMP symbol: {}", name);
            std::_Exit(EXIT_FAILURE);
        }
        return reinterpret_cast<Function>(address);
    }

}

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#include "interceptor_intel.hh"
#elif defined(__clang__)
#include "interceptor_clang.hh"
#elif defined(__GNUC__)
#include "interceptor_gcc.hh"
#else
#include "interceptor_gcc.hh"
#endif