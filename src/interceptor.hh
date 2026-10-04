#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
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
        // Process-wide OPTKIT instance, created on first use and destroyed at exit.
        // It initialises the logger and PMU query state the profilers rely on.
        inline std::unique_ptr<optkit::OPTKIT> &optkit_instance()
        {
            static std::unique_ptr<optkit::OPTKIT> instance;
            return instance;
        }

        inline optkit::OPTKIT &ensure_optkit()
        {
            static std::once_flag once;
            std::call_once(once, []
                           {
#if ORBIT_SNAPSHOT
                               constexpr bool create_folder = true;
                               const std::int64_t khz = utils::requested_cpu_frequency_khz();
                               const bool restore_freq = khz > 0;
#else
                               constexpr bool create_folder = false;
                               const std::int64_t khz = utils::requested_cpu_frequency_khz();
                               const bool restore_freq = true;
#endif
                               optkit_instance().reset(new optkit::OPTKIT{optkit::OPTKIT_CONFIG{create_folder, "", restore_freq}});
                               if (khz > 0)
                               {
                                   try
                                   {
                                       for (int socket = 0; socket < OPTKIT_ENV_CPU_NUM_SOCKETS; ++socket)
                                       {
                                           optkit::frequency::cpu::Frequency::set_core_frequency(khz, socket);
                                       }
                                       std::fprintf(stderr, "ORBIT: CPU frequency set to %lld kHz\n", static_cast<long long>(khz));
                                   }
                                   catch (const std::exception &error)
                                   {
                                       OPTKIT_ERROR("ORBIT: cannot set CPU frequency: {}", error.what());
                                   }
                               } });
            return *optkit_instance();
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
            detail::ensure_optkit();
#if ORBIT_OPTIMIZER
            static std::unordered_map<std::string, Region> region_configs = read_region_configs(utils::optimized_config_path());
            const auto it = region_configs.find(region.name);
            if (it != region_configs.end())
            {
                region.threads = it->second.threads;
                region.chunk = it->second.chunk;
                region.sched = it->second.sched;
                region.frequency = it->second.frequency;
            }
#elif ORBIT_SNAPSHOT
            static const std::int64_t default_freq = utils::requested_cpu_frequency_khz();
            region.frequency = default_freq;
            optkit::pmu::cpu::perf::PerfProfilerConfig perf_config{region.name.c_str(), false /*is_sampling*/};
            perf_config.is_screenshot = true;
            cpu_event_profiler.reset(new optkit::pmu::cpu::perf::BlockProfiler(perf_config, optkit::metrics::performance::cpu_metrics::ai()));
            cpu_energy_profiler.reset(new optkit::energy::rapl::Profiler(
                {region.name.c_str(), "cpu_energy", true, false, optkit::Query::create_folder, !optkit::Query::create_folder},
                optkit::metrics::energy::cpu_metrics::all_metrics()));
#endif
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
#if ORBIT_SNAPSHOT
            region.append_to_file(optkit::utils::EXECUTION_FOLDER_NAME + "/snapshot.conf");
#elif ORBIT_OPTIMIZER
#endif
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