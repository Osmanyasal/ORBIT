#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <omp.h>
#include <string>
#include "optkit.hh"
#include "region.hh"

namespace orbit
{
    static void region_begin(Region &region);
    static void region_end(const Region &region);

    namespace detail
    {
        // "<binary>+0x<offset>" when the caller is inside a loaded object, raw address otherwise.
        inline std::string describe_caller(void *caller)
        {
            char buf[128];
            Dl_info info{};
            if (dladdr(caller, &info) && info.dli_fbase)
            {
                const char *fname = info.dli_fname ? info.dli_fname : "unknown";
                if (const char *slash = std::strrchr(fname, '/'))
                {
                    fname = slash + 1;
                }
                const auto offset = reinterpret_cast<std::uintptr_t>(caller) -
                                    reinterpret_cast<std::uintptr_t>(info.dli_fbase);
                std::snprintf(buf, sizeof(buf), "%s+0x%lx", fname, static_cast<unsigned long>(offset));
            }
            else
            {
                std::snprintf(buf, sizeof(buf), "%p", caller);
            }
            return buf;
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
                     chunk, omp_sched_static, detail::describe_caller(caller)},
              split(legacy),
              parent(active_region)
        {
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
            active_region = parent;
            region_end(region);
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
            std::fprintf(stderr, "ORBIT: cannot resolve OpenMP symbol: %s\n", name);
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