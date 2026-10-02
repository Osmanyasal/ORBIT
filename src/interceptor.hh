#pragma once

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace orbit {

struct Region {
    void (*function)();
    void* caller;
    const char* entry;
    unsigned threads;
    long chunk;
};

static void region_begin(Region& region);
static void region_end(const Region& region);

class RegionScope;
inline thread_local RegionScope* active_region = nullptr;

class RegionScope {
    RegionScope* parent;

public:
    Region region;
    bool split;

        template <typename Function>
        RegionScope(Function function, void* caller, const char* entry,
                unsigned threads, long chunk = 0, bool legacy = false)
                : parent(active_region), region {reinterpret_cast<void (*)()>(function), caller, entry, threads, chunk},
                    split(legacy) {
        region_begin(region);
        if (threads == 1) {
            region.threads = 1;
        }
        active_region = this;
    }

    ~RegionScope() {
        active_region = parent;
        region_end(region);
    }

    RegionScope(const RegionScope&) = delete;
    RegionScope& operator=(const RegionScope&) = delete;
};

template <typename Function>
inline Function resolve(const char* name) {
    void* address = dlsym(RTLD_NEXT, name);
    if (!address) {
        std::fprintf(stderr, "ORBIT: cannot resolve OpenMP symbol: %s\n", name);
        std::_Exit(EXIT_FAILURE);
    }
    return reinterpret_cast<Function>(address);
}

}

#if defined(ORBIT_OPENMP_GCC)
#include "interceptor_gcc.hh"
#elif defined(ORBIT_OPENMP_INTEL)
#include "interceptor_intel.hh"
#elif defined(ORBIT_OPENMP_CLANG)
#include "interceptor_clang.hh"
#else
#include "interceptor_gcc.hh"
#include "interceptor_intel.hh"
#include "interceptor_clang.hh"
#endif