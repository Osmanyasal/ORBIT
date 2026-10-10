#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <omp.h>
#include "policy.hh"
#include "utils.hh"

namespace orbit
{
    struct RegionConfig
    {
        int threads;
        omp_sched_t sched;
        long chunk;
        std::int64_t frequency;
        double time_s;
        double energy_j;
        double edp;
        double total_time_s;
        double total_energy_j;
        double total_edp;

        RegionConfig()
            : threads{0}, sched{static_cast<omp_sched_t>(0)}, chunk{0}, frequency{0},
              time_s{0.0}, energy_j{0.0}, edp{0.0},
              total_time_s{0.0}, total_energy_j{0.0}, total_edp{0.0}
        {
        }

        // A configuration to apply; its measured metrics start at zero.
        RegionConfig(int threads, omp_sched_t sched, long chunk, std::int64_t frequency = 0)
            : threads{threads}, sched{sched}, chunk{chunk}, frequency{frequency},
              time_s{0.0}, energy_j{0.0}, edp{0.0},
              total_time_s{0.0}, total_energy_j{0.0}, total_edp{0.0}
        {
        }

        // Time spent in the region over all its calls, falling back to a single call when unknown.
        double execution_time_s() const { return this->total_time_s > 0.0 ? this->total_time_s : this->time_s; }

        // False for an empty RegionConfig{}, i.e. when no candidate was provided.
        bool is_set() const { return this->threads > 0; }

        // True when both apply the same threads, schedule, chunk and frequency (metrics are ignored).
        bool same_settings(const RegionConfig &other) const
        {
            return this->threads == other.threads && this->sched == other.sched &&
                   this->chunk == other.chunk && this->frequency == other.frequency;
        }
    };

    // A region is its identity plus the configuration currently in effect and, when read from a
    // best_per_region.json file, the candidate configurations that were evaluated.
    class Region
    {
    public:
        void (*function)();
        void *caller; // caller and name can be different due to ASLR
        const char *entry;
        std::string name;

        RegionConfig current; // settings applied to this region and their measured metrics
        RegionConfig fastest;
        RegionConfig best_edp;
        int calls;
        int configs_evaluated;

        Region()
            : function{nullptr}, caller{nullptr}, entry{""}, name{},
              current{}, fastest{}, best_edp{}, calls{0}, configs_evaluated{0}
        {
        }

        Region(void (*function)(), void *caller, const char *entry, std::string name, const RegionConfig &current)
            : function{function}, caller{caller}, entry{entry}, name{std::move(name)},
              current{current}, fastest{}, best_edp{}, calls{0}, configs_evaluated{0}
        {
        }

        // Execution time of the region as evaluated in the configuration file.
        double execution_time_s() const
        {
            const double t = this->fastest.execution_time_s();
            return t > 0.0 ? t : this->best_edp.execution_time_s();
        }

        // Takes over the evaluated candidates of a region read from a configuration file and applies
        // the one selected by the policy as the current configuration. If that candidate is missing
        // the other one is used, and if none exists the current configuration is kept.
        void take_tuning(const Region &other, OptimizePolicy policy)
        {
            this->calls = other.calls;
            this->configs_evaluated = other.configs_evaluated;
            this->fastest = other.fastest;
            this->best_edp = other.best_edp;

            const bool want_fastest = policy == OptimizePolicy::Fastest;
            const RegionConfig &preferred = want_fastest ? this->fastest : this->best_edp;
            const RegionConfig &fallback = want_fastest ? this->best_edp : this->fastest;
            if (preferred.is_set())
                this->current = preferred;
            else if (fallback.is_set())
                this->current = fallback;
        }

        std::string to_json() const
        {
            std::ostringstream ss;
            char caller_buf[32];
            std::snprintf(caller_buf, sizeof(caller_buf), "%p", this->caller);
            char fn_buf[32];
            std::snprintf(fn_buf, sizeof(fn_buf), "%p", reinterpret_cast<void *>(this->function));

            ss << "{\n"
               << "  \"name\": \"" << this->name << "\",\n"
               << "  \"caller\": \"" << caller_buf << "\",\n"
               << "  \"function\": \"" << fn_buf << "\",\n"
               << "  \"entry\": \"" << (this->entry ? this->entry : "") << "\",\n"
               << "  \"threads\": " << this->current.threads << ",\n"
               << "  \"chunk\": " << this->current.chunk << ",\n"
               << "  \"sched\": \"" << utils::sched_to_string(this->current.sched) << "\",\n"
               << "  \"frequency\": " << this->current.frequency << "\n"
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
                out << this->to_json();
            }
        }
    };

    using RegionConfigs = std::unordered_map<std::string, Region>;
}
