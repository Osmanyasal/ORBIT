#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <string>
#include <omp.h>
#include "optkit.hh"

namespace orbit::utils
{
    // Formats the caller address as "<binary>.0x<offset>" when inside a loaded object,
    // or as a hex address otherwise.
    inline std::string describe_caller(void *caller)
    {
        if (!caller)
        {
            return "0x0";
        }

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
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s.0x%lx", fname, static_cast<unsigned long>(offset));
            return buf;
        }

        char buf[32];
        std::snprintf(buf, sizeof(buf), "%p", caller);
        return buf;
    }

    // Parses a frequency string into kHz.
    // If unit is omitted, defaults to MHz (e.g. "2400" -> 2400 MHz -> 2400000 kHz).
    // Explicit units like "2.4GHz" or "2400000KHz" are converted via OPTKIT.
    // Returns kHz on success, or 0 on failure.
    inline std::int64_t parse_cpu_frequency_khz(const std::string &text)
    {
        if (text.empty())
        {
            return 0;
        }

        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            return 0;
        }
        const auto last = text.find_last_not_of(" \t\r\n");
        std::string value = text.substr(first, last - first + 1);

        if (value.find_first_not_of("0123456789.") == std::string::npos)
        {
            value += "MHz";
        }
        try
        {
            const double khz = optkit::frequency::convert_frequency_with_unit(value, optkit::frequency::Unit::KHz);
            if (khz >= 1.0)
            {
                return static_cast<std::int64_t>(khz + 0.5);
            }
        }
        catch (const std::exception &)
        {
        }
        return 0;
    }

    // Reads the CPU frequency from an environment variable (default: "ORBIT_CPU_FREQ").
    // Returns kHz, or 0 when unset or invalid.
    // Reports errors through OPTKIT_ERROR.
    inline std::int64_t requested_cpu_frequency_khz(const char *env_var = "ORBIT_CPU_FREQ")
    {
        const char *text = std::getenv(env_var);
        if (!text || !*text)
        {
            return 0;
        }

        const std::int64_t khz = parse_cpu_frequency_khz(text);
        if (khz <= 0)
        {
            OPTKIT_ERROR("ORBIT: ignoring invalid {}='{}' (expected MHz, e.g. 2400, or with a unit: 2.4GHz)",
                         env_var, text);
            return 0;
        }
        return khz;
    }

    // OpenMP runtime schedule and chunk query helpers
    inline omp_sched_t current_sched()
    {
        omp_sched_t kind;
        int chunk;
        omp_get_schedule(&kind, &chunk);
        return kind;
    }

    inline long current_chunk()
    {
        omp_sched_t kind;
        int chunk;
        omp_get_schedule(&kind, &chunk);
        return chunk;
    }

    // Returns the path of the optimized configuration file read at startup.
    // Checks ORBIT_OPTIMIZED_CONF, then ORBIT_CONFIG, defaulting to "optimized.conf".
    // When the file exists ORBIT applies it; otherwise it runs a snapshot analysis.
    inline const char *optimized_config_path(const char *env_var = "ORBIT_OPTIMIZED_CONF")
    {
        const char *path = std::getenv(env_var);
        if (!path || !*path)
        {
            path = std::getenv("ORBIT_CONFIG");
        }
        return (path && *path) ? path : "optimized.conf";
    }
}
