#pragma once

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <omp.h>
#include "region.hh"

namespace orbit
{
    static void region_begin(Region &region);
    static void region_end(const Region &region);

    class RegionScope;
    inline thread_local RegionScope *active_region = nullptr;

    class RegionScope
    {
        RegionScope *parent;

    public:
        Region region;
        bool split;

        template <typename Function>
        RegionScope(Function function, void *caller, const char *entry,
                    unsigned threads, long chunk = 0, bool legacy = false)
            : parent(active_region), 
              region{reinterpret_cast<void (*)()>(function), caller, entry, static_cast<int>(threads), chunk, omp_sched_static, {}},
              split(legacy)
        {
            Dl_info info;
            if (dladdr(caller, &info) && info.dli_fbase)
            {
                uintptr_t offset = reinterpret_cast<uintptr_t>(caller) -
                                   reinterpret_cast<uintptr_t>(info.dli_fbase);
                const char *fname = info.dli_fname ? info.dli_fname : "unknown";
                const char *slash = std::strrchr(fname, '/');
                if (slash)
                {
                    fname = slash + 1;
                }
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%s+0x%lx", fname, offset);
                region.name = buf;
            }
            else
            {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%p", caller);
                region.name = buf;
            }

            region.threads = !threads ? omp_get_max_threads() : static_cast<int>(threads);
            region_begin(region);
            active_region = this;
        }

        ~RegionScope()
        {
            active_region = parent;
            region_end(region);
        }

        RegionScope(const RegionScope &) = delete;
        RegionScope &operator=(const RegionScope &) = delete;
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