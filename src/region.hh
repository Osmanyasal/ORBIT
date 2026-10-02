#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <mutex>
#include <omp.h>

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

    class Region
    {
    public:
        void (*function)() = nullptr;
        void *caller = nullptr; // caller and name can be different due to ASLR
        const char *entry = "";
        int threads = 0;
        long chunk = 0;
        omp_sched_t sched = static_cast<omp_sched_t>(0);
        std::string name;

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
               << "  \"threads\": " << threads << ",\n"
               << "  \"sched\": \"" << sched_to_string(sched) << "\",\n"
               << "  \"chunk\": " << chunk << "\n"
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

    inline std::unordered_map<std::string, Region> read_region_configs(const std::string &filepath)
    {
        std::unordered_map<std::string, Region> configs;
        std::ifstream in(filepath);
        if (!in.is_open())
        {
            return configs;
        }

        std::string line;
        Region current;
        bool in_object = false;

        auto trim = [](const std::string &s) -> std::string
        {
            auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
                return "";
            auto end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        };

        while (std::getline(in, line))
        {
            std::string trimmed = trim(line);
            if (trimmed.empty())
                continue;

            if (trimmed == "{" || trimmed.rfind('{', 0) == 0)
            {
                current = Region{};
                in_object = true;
                continue;
            }
            if (trimmed == "}" || trimmed == "}," || trimmed.rfind('}', 0) == 0)
            {
                if (in_object && !current.name.empty())
                {
                    configs[current.name] = current;
                }
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
                current.name = str_val;
            else if (key == "caller")
            {
                try
                {
                    current.caller = reinterpret_cast<void *>(std::stoull(str_val, nullptr, 16));
                }
                catch (...)
                {
                }
            }
            else if (key == "function")
            {
                try
                {
                    current.function = reinterpret_cast<void (*)()>(std::stoull(str_val, nullptr, 16));
                }
                catch (...)
                {
                }
            }
            else if (key == "sched")
                current.sched = string_to_sched(str_val);
            else if (key == "threads")
            {
                try
                {
                    current.threads = static_cast<int>(std::stol(val));
                }
                catch (...)
                {
                }
            }
            else if (key == "chunk")
            {
                try
                {
                    current.chunk = std::stol(val);
                }
                catch (...)
                {
                }
            }
        }

        if (in_object && !current.name.empty())
        {
            configs[current.name] = current;
        }

        return configs;
    }

}
