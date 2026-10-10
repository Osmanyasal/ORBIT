#pragma once

#include <cstdint>
#include <string>
#include <omp.h>
#include "utils.hh"
#include "utils/json.hh"

// Helpers for reading values out of nlohmann::json documents.
namespace orbit::utils
{
    // An address stored as a hexadecimal string or as an unsigned number; nullptr otherwise.
    inline void *parse_hex_pointer(const nlohmann::json &j)
    {
        if (j.is_string())
            return parse_hex_address(j.get<std::string>());
        if (j.is_number_unsigned())
            return reinterpret_cast<void *>(j.get<std::uintptr_t>());
        return nullptr;
    }

    // Reads a numeric member that may also be stored as a string. Returns the fallback when the
    // member is missing or cannot be parsed.
    template <typename Number, typename ParseString>
    Number read_json_number(const nlohmann::json &j, const char *key, Number fallback, ParseString parse)
    {
        const auto member = j.find(key);
        if (member == j.end())
            return fallback;
        if (member->is_number())
            return member->template get<Number>();
        if (member->is_string())
        {
            try
            {
                return parse(member->template get<std::string>());
            }
            catch (...)
            {
            }
        }
        return fallback;
    }

    inline double get_json_double(const nlohmann::json &j, const char *key, double fallback = 0.0)
    {
        return read_json_number<double>(j, key, fallback, [](const std::string &s) { return std::stod(s); });
    }

    inline long get_json_long(const nlohmann::json &j, const char *key, long fallback = 0)
    {
        return read_json_number<long>(j, key, fallback, [](const std::string &s) { return std::stol(s); });
    }

    inline int get_json_int(const nlohmann::json &j, const char *key, int fallback = 0)
    {
        return static_cast<int>(get_json_long(j, key, fallback));
    }

    // The schedule is stored by name or as the numeric omp_sched_t value.
    inline omp_sched_t parse_json_sched(const nlohmann::json &sched, omp_sched_t fallback)
    {
        if (sched.is_string())
            return string_to_sched(sched.get<std::string>());
        if (sched.is_number())
            return static_cast<omp_sched_t>(sched.get<int>());
        return fallback;
    }

    // The frequency is stored in kHz as a number, or as text with an optional unit.
    inline std::int64_t parse_json_frequency(const nlohmann::json &frequency, std::int64_t fallback)
    {
        if (frequency.is_number())
            return frequency.get<std::int64_t>();
        if (frequency.is_string())
            return parse_cpu_frequency_khz(frequency.get<std::string>());
        return fallback;
    }
}
