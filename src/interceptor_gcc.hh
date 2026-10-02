#pragma once

extern "C" void GOMP_parallel(void (*function)(void*), void* data,
                              unsigned threads, unsigned flags) {
    using Function = void (*)(void (*)(void*), void*, unsigned, unsigned);
    static const auto next = orbit::resolve<Function>("GOMP_parallel");
    orbit::RegionScope scope(function, __builtin_return_address(0), "GOMP_parallel", threads);
    next(function, data, scope.region.threads, flags);
}

extern "C" void GOMP_parallel_start(void (*function)(void*), void* data, unsigned threads) {
    using Function = void (*)(void (*)(void*), void*, unsigned);
    static const auto next = orbit::resolve<Function>("GOMP_parallel_start");
    auto* scope = new orbit::RegionScope(function, __builtin_return_address(0),
                                         "GOMP_parallel_start", threads, 0, true);
    next(function, data, scope->region.threads);
}

extern "C" void GOMP_parallel_end() {
    using Function = void (*)();
    static const auto next = orbit::resolve<Function>("GOMP_parallel_end");
    auto* region = orbit::active_region;
    next();
    if (region && region->split) {
        delete region;
    }
}

extern "C" void GOMP_parallel_sections(void (*function)(void*), void* data,
                                       unsigned threads, unsigned count, unsigned flags) {
    using Function = void (*)(void (*)(void*), void*, unsigned, unsigned, unsigned);
    static const auto next = orbit::resolve<Function>("GOMP_parallel_sections");
    orbit::RegionScope scope(function, __builtin_return_address(0), "GOMP_parallel_sections", threads);
    next(function, data, scope.region.threads, count, flags);
}

#define ORBIT_PARALLEL_LOOP(API) \
extern "C" void GOMP_parallel_loop_##API(void (*function)(void*), void* data, unsigned threads, \
                                        long start, long end, long increment, long chunk, unsigned flags) { \
    using Function = void (*)(void (*)(void*), void*, unsigned, long, long, long, long, unsigned); \
    static const auto next = orbit::resolve<Function>("GOMP_parallel_loop_" #API); \
    orbit::RegionScope scope(function, __builtin_return_address(0), "GOMP_parallel_loop_" #API, threads, chunk); \
    next(function, data, scope.region.threads, start, end, increment, scope.region.chunk, flags); \
}

ORBIT_PARALLEL_LOOP(static)
ORBIT_PARALLEL_LOOP(dynamic)
ORBIT_PARALLEL_LOOP(guided)
ORBIT_PARALLEL_LOOP(nonmonotonic_dynamic)
ORBIT_PARALLEL_LOOP(nonmonotonic_guided)
#undef ORBIT_PARALLEL_LOOP

#define ORBIT_PARALLEL_RUNTIME(API) \
extern "C" void GOMP_parallel_loop_##API(void (*function)(void*), void* data, unsigned threads, \
                                        long start, long end, long increment, unsigned flags) { \
    using Function = void (*)(void (*)(void*), void*, unsigned, long, long, long, unsigned); \
    static const auto next = orbit::resolve<Function>("GOMP_parallel_loop_" #API); \
    orbit::RegionScope scope(function, __builtin_return_address(0), "GOMP_parallel_loop_" #API, threads); \
    next(function, data, scope.region.threads, start, end, increment, flags); \
}

ORBIT_PARALLEL_RUNTIME(runtime)
ORBIT_PARALLEL_RUNTIME(nonmonotonic_runtime)
ORBIT_PARALLEL_RUNTIME(maybe_nonmonotonic_runtime)
#undef ORBIT_PARALLEL_RUNTIME