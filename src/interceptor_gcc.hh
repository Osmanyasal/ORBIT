#pragma once

extern "C" void GOMP_parallel(void (*function)(void*), void* data,
                              unsigned threads, unsigned flags) {
    using Function = void (*)(void (*)(void*), void*, unsigned, unsigned);
    static const auto next = orbit::resolve<Function>("GOMP_parallel");
    orbit::RegionScope scope(function, __builtin_return_address(0), "GOMP_parallel", threads);
    next(function, data, scope.region.current.threads, flags);
}

extern "C" void GOMP_parallel_start(void (*function)(void*), void* data, unsigned threads) {
    using Function = void (*)(void (*)(void*), void*, unsigned);
    static const auto next = orbit::resolve<Function>("GOMP_parallel_start");
    auto* scope = new orbit::RegionScope(function, __builtin_return_address(0),
                                         "GOMP_parallel_start", threads, 0, true);
    next(function, data, scope->region.current.threads);
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