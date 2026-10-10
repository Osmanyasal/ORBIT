#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <exception>
#include <memory>
#include <omp.h>
#include <string>
#include <unordered_map>
#include "optkit.hh"
#include "config_reader.hh"
#include "policy.hh"
#include "region.hh"
#include "utils.hh"

// Per-call tracing on stdout, compiled in only for Debug builds.
#ifdef ORBIT_MODE_DEBUG
#include <iostream>
#define ORBIT_TRACE(message) (std::cout << message << std::endl)
#else
#define ORBIT_TRACE(message) ((void)0)
#endif

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
            RegionConfigs configs;
            std::int64_t frequency = 0;
            OptimizePolicy policy = OptimizePolicy::BestEdp; // ORBIT_OPTIMIZE_POLICY
        };

        // Process-wide OPTKIT instance, created on first use and destroyed at exit via atexit.
        // It initialises the logger and PMU query state the profilers rely on.
        inline std::unique_ptr<optkit::OPTKIT> &optkit_instance()
        {
            static std::unique_ptr<optkit::OPTKIT> instance;
            return instance;
        }

        inline void release_optkit_instance()
        {
            try
            {
                optkit_instance().reset();
            }
            catch (...)
            {
            }
        }

        // Signal handler for Ctrl+C (SIGINT) and termination (SIGTERM).
        inline void terminate_on_signal(int sig)
        {
            release_optkit_instance();
            // Re-raise or exit with the signal code to preserve standard shell behavior
            std::signal(sig, SIG_DFL);
            ::kill(::getpid(), sig);
        }

        inline void install_signal_handlers()
        {
            struct sigaction sa{};
            sa.sa_handler = terminate_on_signal;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;
            sigaction(SIGINT, &sa, nullptr);
            sigaction(SIGTERM, &sa, nullptr);
        }

        // Chooses the mode: Optimize when ORBIT_OPTIMIZED_CONF names an existing file, Snapshot otherwise.
        inline void load_configuration(Runtime &runtime)
        {
            const char *config_path = utils::optimized_config_path();
            struct stat config_status;
            if (config_path &&
                stat(config_path, &config_status) == 0 &&
                S_ISREG(config_status.st_mode))
            {
                runtime.mode = Mode::Optimize;
                runtime.configs = read_region_configs(config_path);
                runtime.policy = optimize_policy();
                std::fprintf(stderr, "ORBIT: optimize mode, %zu region(s) read from %s, policy %s\n",
                             runtime.configs.size(), config_path, policy_to_string(runtime.policy));
            }
            else
            {
                runtime.mode = Mode::Snapshot;
                std::fprintf(stderr, "ORBIT: %s, snapshot mode active\n",
                             config_path ? "configuration file not found" : "ORBIT_OPTIMIZED_CONF not set");
            }
        }

        inline void start_optkit(const Runtime &runtime)
        {
            const bool is_snapshot = runtime.mode == Mode::Snapshot;
            // Optimize mode changes the frequency per region, so the original must always be restored.
            const bool restore_freq = !is_snapshot || runtime.frequency > 0;
            optkit_instance().reset(new optkit::OPTKIT{optkit::OPTKIT_CONFIG{is_snapshot, "", restore_freq}});

            std::atexit(release_optkit_instance);
            install_signal_handlers();
        }

        inline void apply_requested_frequency(const Runtime &runtime)
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

        inline Runtime load_runtime()
        {
            Runtime runtime;
            runtime.frequency = utils::requested_cpu_frequency_khz();
            load_configuration(runtime);
            start_optkit(runtime);
            if (runtime.frequency > 0)
            {
                apply_requested_frequency(runtime);
            }
            return runtime;
        }

        // Selects the mode, reads the configuration and initialises OPTKIT once, on first use.
        inline const Runtime &ensure_runtime()
        {
            static const Runtime runtime = load_runtime();
            return runtime;
        }

        // What the call site of a parallel region determines, so that it is worked out once instead of
        // on every call.
        class RegionSite
        {
        public:
            std::string name;
            RegionConfig tuned;    // configuration chosen by the policy from the file; not set when none applies
            RegionConfig recorded; // settings last written to snapshot.conf
            bool has_recorded;

            RegionSite(std::string name, const RegionConfig &tuned)
                : name{std::move(name)}, tuned{tuned}, recorded{}, has_recorded{false}
            {
            }
        };

        inline RegionConfig select_tuned_config(const std::string &name, const Runtime &runtime)
        {
            if (runtime.mode != Mode::Optimize)
                return RegionConfig{};

            const auto candidates = runtime.configs.find(name);
            if (candidates == runtime.configs.end())
                return RegionConfig{};

            Region selected{};
            selected.take_tuning(candidates->second, runtime.policy);
            return selected.current;
        }

        // Finds the site of a caller address in a per-thread cache. Resolving a caller needs dladdr and,
        // in Optimize mode, a lookup by name; after the first call from an address only a lookup by
        // pointer remains. Entries are never erased, so references stay valid for the thread's lifetime.
        inline RegionSite &region_site(void *caller)
        {
            thread_local std::unordered_map<void *, RegionSite> sites;

            const auto cached = sites.find(caller);
            if (cached != sites.end())
                return cached->second;

            std::string name = utils::describe_caller(caller);
            const RegionConfig tuned = select_tuned_config(name, ensure_runtime());
            return sites.emplace(caller, RegionSite{std::move(name), tuned}).first->second;
        }
    }

    class RegionScope;
    inline thread_local RegionScope *active_region = nullptr;

    // Brackets one intercepted parallel region: region_begin on construction, region_end on destruction.
    // Scopes nest per thread through active_region; `split` marks heap-allocated scopes whose end is
    // signalled by a separate runtime call (legacy GOMP_parallel_start/end, serialized KMP regions).
    // A scope must end on the thread that started it.
    class RegionScope
    {
    public:
        Region region;
        bool split;

        RegionScope(void (*function)(), void *caller, const char *entry,
                    unsigned threads, long chunk = 0, bool legacy = false)
            : RegionScope(detail::region_site(caller), function, caller, entry, threads, chunk, legacy)
        {
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
            this->cpu_event_profiler.reset();
            this->cpu_energy_profiler.reset();
            active_region = this->parent;
            region_end(this->region);
            if (detail::ensure_runtime().mode == detail::Mode::Snapshot)
            {
                this->record_snapshot();
            }
        }

        RegionScope(const RegionScope &) = delete;
        RegionScope &operator=(const RegionScope &) = delete;

    private:
        RegionScope(detail::RegionSite &site, void (*function)(), void *caller, const char *entry,
                    unsigned threads, long chunk, bool legacy)
            : region(make_region(site, function, caller, entry, threads, chunk)),
              split(legacy),
              parent(active_region),
              cpu_event_profiler{},
              cpu_energy_profiler{},
              site(site)
        {
            const detail::Runtime &runtime = detail::ensure_runtime();
            if (runtime.mode == detail::Mode::Snapshot)
            {
                this->start_profilers(runtime);
            }

            ORBIT_TRACE(this->region.to_json());

            region_begin(this->region);
            active_region = this;
        }

        // The configuration from the file when the site has one, otherwise the settings in effect.
        static Region make_region(const detail::RegionSite &site, void (*function)(), void *caller,
                                  const char *entry, unsigned threads, long chunk)
        {
            if (site.tuned.is_set())
            {
                return Region{function, caller, entry, site.name, site.tuned};
            }

            const utils::Schedule schedule = utils::current_schedule();
            return Region{function, caller, entry, site.name,
                          RegionConfig{threads ? static_cast<int>(threads) : omp_get_max_threads(),
                                       schedule.kind,
                                       chunk ? chunk : schedule.chunk}};
        }

        void start_profilers(const detail::Runtime &runtime)
        {
            this->region.current.frequency = runtime.frequency; // set the region frequency to the runtime frequency in snapshot mode
            optkit::pmu::cpu::perf::PerfProfilerConfig perf_config{this->region.name.c_str(), false /*is_sampling*/};

            auto metrics = optkit::metrics::performance::cpu_metrics::ipc();
            metrics.add(optkit::metrics::performance::cpu_metrics::l2_hit_ratio());
            metrics.add(optkit::metrics::performance::cpu_metrics::l3_mpki());
            metrics.add(optkit::metrics::performance::cpu_metrics::branch_mispr_ratio());

            this->cpu_event_profiler.reset(new optkit::pmu::cpu::perf::BlockProfiler(perf_config, metrics));
            this->cpu_energy_profiler.reset(new optkit::energy::rapl::Profiler(
                {this->region.name.c_str(), "cpu_energy", true, false, optkit::Query::create_folder, !optkit::Query::create_folder},
                optkit::metrics::energy::cpu_metrics::all_metrics()));
        }

        // Appends the region to snapshot.conf, except when its site already recorded the same settings:
        // readers keep one entry per region name, so repeating it only costs a file write per call.
        void record_snapshot()
        {
            if (this->site.has_recorded && this->site.recorded.same_settings(this->region.current))
            {
                return;
            }
            this->region.append_to_file(optkit::utils::EXECUTION_FOLDER_NAME + "/snapshot.conf");
            this->site.recorded = this->region.current;
            this->site.has_recorded = true;
        }

        RegionScope *parent;
        std::unique_ptr<optkit::pmu::cpu::perf::BlockProfiler> cpu_event_profiler;
        std::unique_ptr<optkit::energy::rapl::Profiler> cpu_energy_profiler;
        detail::RegionSite &site;
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
