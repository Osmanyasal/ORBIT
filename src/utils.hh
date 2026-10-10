#pragma once

#include <cstddef>
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
    // Returns the text without leading and trailing whitespace.
    inline std::string trim(const std::string &text)
    {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            return "";
        }
        const auto last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1);
    }

    // Removes one pair of surrounding double quotes, if present.
    inline std::string strip_quotes(const std::string &text)
    {
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
        {
            return text.substr(1, text.size() - 2);
        }
        return text;
    }

    inline bool has_suffix(const std::string &text, const char *suffix)
    {
        const std::size_t length = std::strlen(suffix);
        return text.size() >= length && text.compare(text.size() - length, length, suffix) == 0;
    }

    // Runs the assignment and ignores a value that cannot be converted, keeping the field's default.
    template <typename Assign>
    void assign_if_valid(Assign assign)
    {
        try
        {
            assign();
        }
        catch (...)
        {
        }
    }

    // Parses a hexadecimal address, with or without 0x; nullptr when empty or invalid.
    inline void *parse_hex_address(const std::string &text)
    {
        if (text.empty())
        {
            return nullptr;
        }
        try
        {
            return reinterpret_cast<void *>(std::stoull(text, nullptr, 16));
        }
        catch (...)
        {
            return nullptr;
        }
    }

    // Parses a duration such as "100ms" or "0.1s". A bare number is seconds, or milliseconds when
    // it is 10 or more. Returns the fallback when the text is empty or invalid.
    inline double parse_duration_s(const std::string &text, double fallback)
    {
        const std::string s = trim(text);
        if (s.empty())
        {
            return fallback;
        }

        try
        {
            if (s.size() > 2 && (has_suffix(s, "ms") || has_suffix(s, "MS")))
            {
                return std::stod(s.substr(0, s.size() - 2)) / 1000.0;
            }
            if (s.size() > 1 && (s.back() == 's' || s.back() == 'S'))
            {
                return std::stod(s.substr(0, s.size() - 1));
            }

            const double value = std::stod(s);
            return value >= 10.0 ? value / 1000.0 : value;
        }
        catch (...)
        {
            return fallback;
        }
    }

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
        std::string value = trim(text);
        if (value.empty())
        {
            return 0;
        }

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

    // The OpenMP runtime schedule in effect (run-sched-var).
    struct Schedule
    {
        omp_sched_t kind;
        long chunk;
    };

    // Queries the runtime once; both fields come from the same omp_get_schedule call.
    inline Schedule current_schedule()
    {
        omp_sched_t kind;
        int chunk;
        omp_get_schedule(&kind, &chunk);
        // libgomp may OR a monotonic/nonmonotonic modifier into the kind (e.g. 0x80000001 for "static,N"),
        // which would not match any plain schedule kind.
        constexpr unsigned modifier_bits = 0x80000000u | 0x40000000u;
        return {static_cast<omp_sched_t>(static_cast<unsigned>(kind) & ~modifier_bits), chunk};
    }

    inline const char *sched_to_string(omp_sched_t sched)
    {
        switch (sched)
        {
        case omp_sched_static:
            return "static";
        case omp_sched_dynamic:
            return "dynamic";
        case omp_sched_guided:
            return "guided";
        case omp_sched_auto:
            return "auto";
        default:
            return "unknown";
        }
    }

    // Unknown names give 0, the same value an unset schedule has.
    inline omp_sched_t string_to_sched(const std::string &str)
    {
        if (str == "static")
            return omp_sched_static;
        if (str == "dynamic")
            return omp_sched_dynamic;
        if (str == "guided")
            return omp_sched_guided;
        if (str == "auto")
            return omp_sched_auto;
        return static_cast<omp_sched_t>(0);
    }

    // Returns the optimized configuration file path from ORBIT_OPTIMIZED_CONF, or nullptr when unset or empty.
    // When it names an existing file ORBIT applies it; otherwise it runs a snapshot analysis.
    inline const char *optimized_config_path(const char *env_var = "ORBIT_OPTIMIZED_CONF")
    {
        const char *path = std::getenv(env_var);
        return (path && *path) ? path : nullptr;
    }
}
