#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <mutex>
#include <omp.h>
#include "utils.hh"
#include "utils/json.hh"

namespace orbit
{

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

    struct RegionConfig
    {
        int threads = 0;
        omp_sched_t sched = static_cast<omp_sched_t>(0);
        long chunk = 0;
        std::int64_t frequency = 0;
        double time_s = 0.0;
        double energy_j = 0.0;
        double edp = 0.0;
        double total_time_s = 0.0;
        double total_energy_j = 0.0;
        double total_edp = 0.0;

        // Time spent in the region over all its calls, falling back to a single call when unknown.
        double execution_time_s() const { return total_time_s > 0.0 ? total_time_s : time_s; }
    };

    // A region is its identity plus the configuration currently in effect and, when read from a
    // best_per_region.json file, the candidate configurations that were evaluated.
    class Region
    {
    public:
        void (*function)() = nullptr;
        void *caller = nullptr; // caller and name can be different due to ASLR
        const char *entry = "";
        std::string name;

        RegionConfig current{}; // settings applied to this region and their measured metrics
        int calls = 0;
        int configs_evaluated = 0;
        RegionConfig fastest{};
        RegionConfig best_edp{};

        Region() = default;

        Region(void (*function)(), void *caller, const char *entry,
               int threads, long chunk, omp_sched_t sched, std::string name,
               std::int64_t frequency = 0)
            : function(function), caller(caller), entry(entry), name(std::move(name)),
              current{threads, sched, chunk, frequency}
        {
        }

        // Takes over the tuned configuration and the evaluated candidates of a region read from a
        // configuration file, keeping this region's identity.
        void take_tuning(const Region &other)
        {
            current = other.current;
            calls = other.calls;
            configs_evaluated = other.configs_evaluated;
            fastest = other.fastest;
            best_edp = other.best_edp;
        }

        std::string to_json() const
        {
            std::ostringstream ss;
            char caller_buf[32];
            std::snprintf(caller_buf, sizeof(caller_buf), "%p", caller);
            char fn_buf[32];
            std::snprintf(fn_buf, sizeof(fn_buf), "%p", reinterpret_cast<void *>(function));

            ss << "{\n"
               << "  \"name\": \"" << name << "\",\n"
               << "  \"caller\": \"" << caller_buf << "\",\n"
               << "  \"function\": \"" << fn_buf << "\",\n"
               << "  \"entry\": \"" << (entry ? entry : "") << "\",\n"
               << "  \"threads\": " << current.threads << ",\n"
               << "  \"chunk\": " << current.chunk << ",\n"
               << "  \"sched\": \"" << sched_to_string(current.sched) << "\",\n"
               << "  \"frequency\": " << current.frequency << "\n"
               << "}\n";
            return ss.str();
        }

        void append_to_file(const std::string &filepath) const
        {
            static std::mutex file_mutex;
            std::lock_guard<std::mutex> lock(file_mutex);
            std::ofstream out(filepath, std::ios::app);
            if (out.is_open())
            {
                out << to_json();
            }
        }
    };

    namespace detail
    {
        inline void *parse_hex_pointer(const nlohmann::json &j)
        {
            if (j.is_string())
            {
                std::string s = j.get<std::string>();
                if (!s.empty())
                {
                    try
                    {
                        return reinterpret_cast<void *>(std::stoull(s, nullptr, 16));
                    }
                    catch (...)
                    {
                    }
                }
            }
            else if (j.is_number_unsigned())
            {
                return reinterpret_cast<void *>(j.get<std::uintptr_t>());
            }
            return nullptr;
        }

        inline double get_json_double(const nlohmann::json &j, const char *key, double fallback = 0.0)
        {
            if (!j.contains(key))
                return fallback;
            const auto &val = j[key];
            if (val.is_number())
                return val.get<double>();
            if (val.is_string())
            {
                try
                {
                    return std::stod(val.get<std::string>());
                }
                catch (...)
                {
                }
            }
            return fallback;
        }

        inline long get_json_long(const nlohmann::json &j, const char *key, long fallback = 0)
        {
            if (!j.contains(key))
                return fallback;
            const auto &val = j[key];
            if (val.is_number())
                return val.get<long>();
            if (val.is_string())
            {
                try
                {
                    return std::stol(val.get<std::string>());
                }
                catch (...)
                {
                }
            }
            return fallback;
        }

        inline int get_json_int(const nlohmann::json &j, const char *key, int fallback = 0)
        {
            return static_cast<int>(get_json_long(j, key, fallback));
        }

        inline RegionConfig parse_region_config(const nlohmann::json &j)
        {
            RegionConfig cfg;
            if (!j.is_object())
                return cfg;

            cfg.threads = get_json_int(j, "threads", 0);
            if (j.contains("sched"))
            {
                if (j["sched"].is_string())
                    cfg.sched = string_to_sched(j["sched"].get<std::string>());
                else if (j["sched"].is_number())
                    cfg.sched = static_cast<omp_sched_t>(j["sched"].get<int>());
            }
            cfg.chunk = get_json_long(j, "chunk", 0);
            if (j.contains("frequency"))
            {
                if (j["frequency"].is_number())
                    cfg.frequency = j["frequency"].get<std::int64_t>();
                else if (j["frequency"].is_string())
                    cfg.frequency = utils::parse_cpu_frequency_khz(j["frequency"].get<std::string>());
            }
            cfg.time_s = get_json_double(j, "time_s", 0.0);
            cfg.energy_j = get_json_double(j, "energy_j", 0.0);
            cfg.edp = get_json_double(j, "edp", 0.0);
            cfg.total_time_s = get_json_double(j, "total_time_s", 0.0);
            cfg.total_energy_j = get_json_double(j, "total_energy_j", 0.0);
            cfg.total_edp = get_json_double(j, "total_edp", 0.0);

            return cfg;
        }

        inline Region parse_region_entry(const std::string &name, const nlohmann::json &j, const std::string &policy)
        {
            Region r;
            r.name = name;
            if (j.contains("caller"))
                r.caller = parse_hex_pointer(j["caller"]);
            if (j.contains("function"))
                r.function = reinterpret_cast<void (*)()>(parse_hex_pointer(j["function"]));
            r.calls = get_json_int(j, "calls", 0);
            r.configs_evaluated = get_json_int(j, "configs_evaluated", 0);

            const bool has_fastest = j.contains("fastest") && j["fastest"].is_object();
            const bool has_edp = j.contains("best_edp") && j["best_edp"].is_object();

            if (has_fastest)
                r.fastest = parse_region_config(j["fastest"]);
            if (has_edp)
                r.best_edp = parse_region_config(j["best_edp"]);

            const bool prefer_edp = (policy == "edp" || policy == "best_edp");
            if (has_edp && (prefer_edp || !has_fastest))
                r.current = r.best_edp;
            else if (has_fastest)
                r.current = r.fastest;
            else
                r.current = parse_region_config(j); // flat configuration object

            return r;
        }

        inline double parse_time_threshold_env(double fallback = 0.1)
        {
            const char *env = std::getenv("ORBIT_TIME_THRESHOLD");
            if (!env || !*env)
                env = std::getenv("ORBIT_REGION_THRESHOLD");
            if (!env || !*env)
                env = std::getenv("ORBIT_MIN_REGION_TIME");
            if (!env || !*env)
                return fallback;

            std::string s(env);
            auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
                return fallback;
            auto end = s.find_last_not_of(" \t\r\n");
            s = s.substr(start, end - start + 1);

            try
            {
                if (s.size() > 2 && (s.substr(s.size() - 2) == "ms" || s.substr(s.size() - 2) == "MS"))
                {
                    return std::stod(s.substr(0, s.size() - 2)) / 1000.0;
                }
                if (s.size() > 1 && (s.back() == 's' || s.back() == 'S'))
                {
                    return std::stod(s.substr(0, s.size() - 1));
                }
                double val = std::stod(s);
                // If a value >= 10 without unit was given, interpret as milliseconds
                if (val >= 10.0)
                {
                    return val / 1000.0;
                }
                return val;
            }
            catch (...)
            {
                return fallback;
            }
        }

        inline std::string get_effective_policy(const std::string &policy)
        {
            if (!policy.empty())
                return policy;
            const char *env = std::getenv("ORBIT_OPTIMIZE_POLICY");
            if (!env || !*env)
                env = std::getenv("ORBIT_POLICY");
            if (!env || !*env)
                env = std::getenv("ORBIT_TARGET");
            if (env && *env)
                return std::string(env);
            return "fastest";
        }

        inline bool should_include_region(const Region &r, double min_execution_time_s, const nlohmann::json &j)
        {
            if (min_execution_time_s <= 0.0)
                return true;

            const double exec_time = r.current.execution_time_s();

            if (exec_time > 0.0)
            {
                return exec_time > min_execution_time_s;
            }

            bool has_timing = j.contains("fastest") || j.contains("best_edp") ||
                              j.contains("total_time_s") || j.contains("time_s");
            if (has_timing)
            {
                return false;
            }

            // Legacy configuration without timing data is preserved
            return true;
        }
    }

    // Reads region configurations from file.
    // min_execution_time_s: minimum execution time in seconds (default: 0.1s = 100ms; set <= 0 to disable).
    // Can also be configured via ORBIT_TIME_THRESHOLD (e.g. "100ms", "0.1s").
    // policy: optimization objective ("fastest" or "edp"/"best_edp"; can also be set via ORBIT_OPTIMIZE_POLICY).
    inline std::unordered_map<std::string, Region> read_region_configs(
        const std::string &filepath,
        double min_execution_time_s = 0.1,
        const std::string &policy = "")
    {
        std::unordered_map<std::string, Region> configs;
        std::ifstream in(filepath);
        if (!in.is_open())
        {
            return configs;
        }

        double effective_threshold = min_execution_time_s;
        if (min_execution_time_s > 0.0 &&
            (std::getenv("ORBIT_TIME_THRESHOLD") || std::getenv("ORBIT_REGION_THRESHOLD") || std::getenv("ORBIT_MIN_REGION_TIME")))
        {
            effective_threshold = detail::parse_time_threshold_env(min_execution_time_s);
        }

        std::string effective_policy = detail::get_effective_policy(policy);

        std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

        try
        {
            auto root = nlohmann::json::parse(content);
            if (root.is_object())
            {
                if (root.contains("name") && root["name"].is_string())
                {
                    // Single region object
                    std::string name = root["name"].get<std::string>();
                    Region r = detail::parse_region_entry(name, root, effective_policy);
                    if (detail::should_include_region(r, effective_threshold, root))
                    {
                        configs[r.name] = r;
                    }
                    return configs;
                }

                for (auto &[key, val] : root.items())
                {
                    if (key.rfind('_', 0) == 0 || !val.is_object())
                        continue;
                    Region r = detail::parse_region_entry(key, val, effective_policy);
                    if (detail::should_include_region(r, effective_threshold, val))
                    {
                        configs[r.name] = r;
                    }
                }
                return configs;
            }
            else if (root.is_array())
            {
                for (auto &item : root)
                {
                    if (!item.is_object())
                        continue;
                    std::string name = item.value("name", "");
                    if (name.empty())
                        continue;
                    Region r = detail::parse_region_entry(name, item, effective_policy);
                    if (detail::should_include_region(r, effective_threshold, item))
                    {
                        configs[r.name] = r;
                    }
                }
                return configs;
            }
        }
        catch (const nlohmann::json::parse_error &)
        {
            // Fallback for concatenated JSON objects (e.g. legacy snapshot.conf)
        }

        std::istringstream stream(content);
        std::string line;
        Region parsed;
        bool in_object = false;

        auto trim = [](const std::string &s) -> std::string
        {
            auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
                return "";
            auto end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        };

        auto flush_parsed = [&]() {
            if (in_object && !parsed.name.empty())
            {
                const double exec_time = parsed.current.execution_time_s();
                if (effective_threshold <= 0.0 || exec_time <= 0.0 || exec_time > effective_threshold)
                {
                    configs[parsed.name] = parsed;
                }
            }
        };

        while (std::getline(stream, line))
        {
            std::string trimmed = trim(line);
            if (trimmed.empty())
                continue;

            if (trimmed == "{" || trimmed.rfind('{', 0) == 0)
            {
                parsed = Region{};
                in_object = true;
                continue;
            }
            if (trimmed == "}" || trimmed == "}," || trimmed.rfind('}', 0) == 0)
            {
                flush_parsed();
                in_object = false;
                continue;
            }

            if (!in_object)
                continue;

            auto colon = trimmed.find(':');
            if (colon == std::string::npos)
                continue;

            std::string key = trim(trimmed.substr(0, colon));
            std::string val = trim(trimmed.substr(colon + 1));

            // remove surrounding quotes from key
            if (key.size() >= 2 && key.front() == '"' && key.back() == '"')
                key = key.substr(1, key.size() - 2);

            // remove trailing comma from val if any
            if (!val.empty() && val.back() == ',')
                val.pop_back();
            val = trim(val);

            // remove quotes from string values
            std::string str_val = val;
            if (str_val.size() >= 2 && str_val.front() == '"' && str_val.back() == '"')
                str_val = str_val.substr(1, str_val.size() - 2);

            if (key == "name")
                parsed.name = str_val;
            else if (key == "caller")
            {
                try
                {
                    parsed.caller = reinterpret_cast<void *>(std::stoull(str_val, nullptr, 16));
                }
                catch (...)
                {
                }
            }
            else if (key == "function")
            {
                try
                {
                    parsed.function = reinterpret_cast<void (*)()>(std::stoull(str_val, nullptr, 16));
                }
                catch (...)
                {
                }
            }
            else if (key == "sched")
                parsed.current.sched = string_to_sched(str_val);
            else if (key == "threads")
            {
                try
                {
                    parsed.current.threads = static_cast<int>(std::stol(val));
                }
                catch (...)
                {
                }
            }
            else if (key == "chunk")
            {
                try
                {
                    parsed.current.chunk = std::stol(val);
                }
                catch (...)
                {
                }
            }
            else if (key == "frequency")
            {
                if (val.find_first_not_of("0123456789") == std::string::npos)
                {
                    try
                    {
                        parsed.current.frequency = std::stoll(val);
                    }
                    catch (...)
                    {
                    }
                }
                else
                {
                    parsed.current.frequency = utils::parse_cpu_frequency_khz(str_val);
                }
            }
            else if (key == "calls")
            {
                try
                {
                    parsed.calls = static_cast<int>(std::stol(val));
                }
                catch (...)
                {
                }
            }
            else if (key == "time_s")
            {
                try
                {
                    parsed.current.time_s = std::stod(val);
                }
                catch (...)
                {
                }
            }
            else if (key == "total_time_s")
            {
                try
                {
                    parsed.current.total_time_s = std::stod(val);
                }
                catch (...)
                {
                }
            }
        }

        flush_parsed();

        return configs;
    }

}

