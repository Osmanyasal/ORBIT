#pragma once

#include <climits>
#include <cstdarg>
#include <cstdint>
#include <vector>
#include <ffi.h>

namespace orbit::kmp {

struct Location {
    std::int32_t reserved1;
    std::int32_t flags;
    std::int32_t reserved2;
    std::int32_t reserved3;
    const char* source;
};

using Microtask = void (*)(std::int32_t*, std::int32_t*, ...);
inline thread_local unsigned pending_threads = 0;

}

extern "C" void __kmpc_push_num_threads(orbit::kmp::Location* location,
                                       std::int32_t global_thread, std::int32_t threads) {
    using Function = void (*)(orbit::kmp::Location*, std::int32_t, std::int32_t);
    static const auto next = orbit::resolve<Function>("__kmpc_push_num_threads");
    orbit::kmp::pending_threads = threads > 0 ? static_cast<unsigned>(threads) : 0;
    next(location, global_thread, threads);
}

extern "C" void __kmpc_fork_call(orbit::kmp::Location* location, std::int32_t argc,
                                orbit::kmp::Microtask function, ...) {
    if (argc < 0) {
        OPTKIT_ERROR("ORBIT: negative captured-argument count");
        std::_Exit(EXIT_FAILURE);
    }
    using Function = void (*)(orbit::kmp::Location*, std::int32_t, orbit::kmp::Microtask, ...);
    static const auto next = orbit::resolve<Function>("__kmpc_fork_call");
    const auto total = static_cast<unsigned>(argc) + 3;
    std::vector<void*> captured(static_cast<unsigned>(argc));
    std::vector<void*> values(total);
    std::vector<ffi_type*> types(total, &ffi_type_pointer);
    types[1] = &ffi_type_sint32;
    values[0] = &location;
    values[1] = &argc;
    values[2] = &function;
    va_list arguments;
    va_start(arguments, function);
    for (std::int32_t index = 0; index < argc; ++index) {
        captured[index] = va_arg(arguments, void*);
        values[static_cast<unsigned>(index) + 3] = &captured[index];
    }
    va_end(arguments);
    ffi_cif interface;
    if (ffi_prep_cif_var(&interface, FFI_DEFAULT_ABI, 3, total,
                         &ffi_type_void, types.data()) != FFI_OK) {
        OPTKIT_ERROR("ORBIT: cannot prepare OpenMP fork call");
        std::_Exit(EXIT_FAILURE);
    }
    const unsigned requested_threads = orbit::kmp::pending_threads;
    orbit::kmp::pending_threads = 0;
    orbit::RegionScope scope(function, __builtin_return_address(0), "__kmpc_fork_call", requested_threads);
    if (scope.region.threads > INT_MAX) {
        OPTKIT_ERROR("ORBIT: thread count exceeds KMP ABI range");
        std::_Exit(EXIT_FAILURE);
    }
    if (scope.region.threads && scope.region.threads != requested_threads) {
        using GetThread = std::int32_t (*)(orbit::kmp::Location*);
        using PushThreads = void (*)(orbit::kmp::Location*, std::int32_t, std::int32_t);
        static const auto get_thread = orbit::resolve<GetThread>("__kmpc_global_thread_num");
        static const auto push_threads = orbit::resolve<PushThreads>("__kmpc_push_num_threads");
        push_threads(location, get_thread(location), static_cast<std::int32_t>(scope.region.threads));
    }
    ffi_call(&interface, FFI_FN(next), nullptr, values.data());
}

extern "C" void __kmpc_serialized_parallel(orbit::kmp::Location* location, std::int32_t global_thread) {
    using Function = void (*)(orbit::kmp::Location*, std::int32_t);
    static const auto next = orbit::resolve<Function>("__kmpc_serialized_parallel");
    orbit::kmp::pending_threads = 0;
    new orbit::RegionScope(static_cast<void (*)()>(nullptr), __builtin_return_address(0),
                           "__kmpc_serialized_parallel", 1, 0, true);
    next(location, global_thread);
}

extern "C" void __kmpc_end_serialized_parallel(orbit::kmp::Location* location,
                                             std::int32_t global_thread) {
    using Function = void (*)(orbit::kmp::Location*, std::int32_t);
    static const auto next = orbit::resolve<Function>("__kmpc_end_serialized_parallel");
    auto* scope = orbit::active_region;
    next(location, global_thread);
    if (scope && scope->split) {
        delete scope;
    }
}