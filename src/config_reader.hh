#pragma once

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include "json_utils.hh"
#include "region.hh"
#include "utils.hh"
#include "utils/json.hh"

// Reading region configurations from a best_per_region.json file or a snapshot.conf.
namespace orbit
{
    namespace detail
    {
        // ---- Environment ---------------------------------------------------------------------

        // First non-empty value of the accepted threshold variables, or nullptr.
        inline const char *time_threshold_env()
        {
            for (const char *name : {"ORBIT_TIME_THRESHOLD", "ORBIT_REGION_THRESHOLD", "ORBIT_MIN_REGION_TIME"})
            {
                const char *value = std::getenv(name);
                if (value && *value)
                    return value;
            }
            return nullptr;
        }

        // The minimum execution time a region needs to be kept: the environment overrides the default,
        // unless the default is <= 0, which disables the filter.
        inline double effective_min_execution_time_s(double default_s)
        {
            if (default_s <= 0.0)
                return default_s;
            const char *text = time_threshold_env();
            return text ? utils::parse_duration_s(text, default_s) : default_s;
        }

        // ---- JSON configuration --------------------------------------------------------------

        inline RegionConfig parse_region_config(const nlohmann::json &j)
        {
            RegionConfig cfg;
            if (!j.is_object())
                return cfg;

            cfg.threads = utils::get_json_int(j, "threads", 0);
            if (j.contains("sched"))
                cfg.sched = utils::parse_json_sched(j["sched"], cfg.sched);
            cfg.chunk = utils::get_json_long(j, "chunk", 0);
            if (j.contains("frequency"))
                cfg.frequency = utils::parse_json_frequency(j["frequency"], cfg.frequency);
            cfg.time_s = utils::get_json_double(j, "time_s", 0.0);
            cfg.energy_j = utils::get_json_double(j, "energy_j", 0.0);
            cfg.edp = utils::get_json_double(j, "edp", 0.0);
            cfg.total_time_s = utils::get_json_double(j, "total_time_s", 0.0);
            cfg.total_energy_j = utils::get_json_double(j, "total_energy_j", 0.0);
            cfg.total_edp = utils::get_json_double(j, "total_edp", 0.0);

            return cfg;
        }

        inline Region parse_region_entry(const std::string &name, const nlohmann::json &j)
        {
            Region r{};
            r.name = name;
            if (j.contains("caller"))
                r.caller = utils::parse_hex_pointer(j["caller"]);
            if (j.contains("function"))
                r.function = reinterpret_cast<void (*)()>(utils::parse_hex_pointer(j["function"]));
            r.calls = utils::get_json_int(j, "calls", 0);
            r.configs_evaluated = utils::get_json_int(j, "configs_evaluated", 0);

            const bool has_fastest = j.contains("fastest") && j["fastest"].is_object();
            const bool has_edp = j.contains("best_edp") && j["best_edp"].is_object();

            // r.current stays empty; the policy picks from the candidates when the region runs.
            if (has_fastest)
                r.fastest = parse_region_config(j["fastest"]);
            if (has_edp)
                r.best_edp = parse_region_config(j["best_edp"]);
            if (!has_fastest && !has_edp)
                r.fastest = r.best_edp = parse_region_config(j); // flat configuration object

            return r;
        }

        inline bool should_include_region(const Region &r, double min_execution_time_s, const nlohmann::json &j)
        {
            if (min_execution_time_s <= 0.0)
                return true;

            const double exec_time = r.execution_time_s();

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

        inline void add_json_region(RegionConfigs &configs, Region region, double min_execution_time_s,
                                    const nlohmann::json &j)
        {
            if (!should_include_region(region, min_execution_time_s, j))
                return;
            const std::string name = region.name;
            configs[name] = std::move(region);
        }

        // Accepts a single region object, an object mapping region names to regions (keys starting
        // with '_' are metadata), or an array of region objects. Returns false when the text is not
        // such a JSON document.
        inline bool parse_json_configs(const std::string &content, double min_execution_time_s, RegionConfigs &configs)
        {
            const nlohmann::json root = nlohmann::json::parse(content, nullptr, false);

            if (root.is_object())
            {
                if (root.contains("name") && root["name"].is_string())
                {
                    add_json_region(configs, parse_region_entry(root["name"].get<std::string>(), root),
                                    min_execution_time_s, root);
                    return true;
                }

                for (auto &[key, val] : root.items())
                {
                    if (key.rfind('_', 0) == 0 || !val.is_object())
                        continue;
                    add_json_region(configs, parse_region_entry(key, val), min_execution_time_s, val);
                }
                return true;
            }

            if (root.is_array())
            {
                for (const auto &item : root)
                {
                    if (!item.is_object())
                        continue;
                    const std::string name = item.value("name", "");
                    if (name.empty())
                        continue;
                    add_json_region(configs, parse_region_entry(name, item), min_execution_time_s, item);
                }
                return true;
            }

            return false;
        }

        // ---- Legacy snapshot.conf ------------------------------------------------------------
        // A snapshot.conf is a concatenation of JSON objects, one per recorded region, which is not a
        // valid JSON document. Its objects are read line by line ("key": value).

        // Applies one `"key": value` line to the region; unknown keys are ignored.
        inline void apply_legacy_field(Region &region, const std::string &raw_key, const std::string &raw_value)
        {
            const std::string key = utils::strip_quotes(utils::trim(raw_key));

            std::string value = utils::trim(raw_value);
            if (!value.empty() && value.back() == ',')
                value.pop_back();
            value = utils::trim(value);
            const std::string text = utils::strip_quotes(value);

            if (key == "name")
                region.name = text;
            else if (key == "caller")
                region.caller = utils::parse_hex_address(text);
            else if (key == "function")
                region.function = reinterpret_cast<void (*)()>(utils::parse_hex_address(text));
            else if (key == "sched")
                region.current.sched = utils::string_to_sched(text);
            else if (key == "threads")
                utils::assign_if_valid([&] { region.current.threads = static_cast<int>(std::stol(value)); });
            else if (key == "chunk")
                utils::assign_if_valid([&] { region.current.chunk = std::stol(value); });
            else if (key == "frequency")
            {
                if (value.find_first_not_of("0123456789") == std::string::npos)
                    utils::assign_if_valid([&] { region.current.frequency = std::stoll(value); });
                else
                    region.current.frequency = utils::parse_cpu_frequency_khz(text);
            }
            else if (key == "calls")
                utils::assign_if_valid([&] { region.calls = static_cast<int>(std::stol(value)); });
            else if (key == "time_s")
                utils::assign_if_valid([&] { region.current.time_s = std::stod(value); });
            else if (key == "total_time_s")
                utils::assign_if_valid([&] { region.current.total_time_s = std::stod(value); });
        }

        inline void parse_legacy_configs(const std::string &content, double min_execution_time_s, RegionConfigs &configs)
        {
            std::istringstream stream(content);
            std::string line;
            Region parsed{};
            bool in_object = false;

            // Legacy objects hold a single flat configuration; it becomes the only candidate.
            auto flush_parsed = [&]()
            {
                if (!in_object || parsed.name.empty())
                    return;
                parsed.fastest = parsed.best_edp = parsed.current;
                parsed.current = RegionConfig{};
                const double exec_time = parsed.execution_time_s();
                if (min_execution_time_s <= 0.0 || exec_time <= 0.0 || exec_time > min_execution_time_s)
                    configs[parsed.name] = parsed;
            };

            while (std::getline(stream, line))
            {
                const std::string trimmed = utils::trim(line);
                if (trimmed.empty())
                    continue;

                if (trimmed.front() == '{')
                {
                    parsed = Region{};
                    in_object = true;
                    continue;
                }
                if (trimmed.front() == '}')
                {
                    flush_parsed();
                    in_object = false;
                    continue;
                }

                const auto colon = trimmed.find(':');
                if (!in_object || colon == std::string::npos)
                    continue;
                apply_legacy_field(parsed, trimmed.substr(0, colon), trimmed.substr(colon + 1));
            }

            flush_parsed();
        }
    }

    // Reads region configurations from file. Only the evaluated candidates (fastest, best_edp) are
    // filled in; Region::current is left empty and chosen later with Region::take_tuning.
    // min_execution_time_s: minimum execution time in seconds (default: 0.1s = 100ms; set <= 0 to disable).
    // Can also be configured via ORBIT_TIME_THRESHOLD (e.g. "100ms", "0.1s").
    inline RegionConfigs read_region_configs(const std::string &filepath, double min_execution_time_s = 0.1)
    {
        RegionConfigs configs;
        std::ifstream in(filepath);
        if (!in.is_open())
        {
            return configs;
        }

        const double threshold = detail::effective_min_execution_time_s(min_execution_time_s);
        const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

        if (!detail::parse_json_configs(content, threshold, configs))
            detail::parse_legacy_configs(content, threshold, configs);

        return configs;
    }
}
