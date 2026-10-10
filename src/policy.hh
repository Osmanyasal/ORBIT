#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace orbit
{
    // Which evaluated candidate of a region is applied as its current configuration.
    enum class OptimizePolicy
    {
        BestEdp, // lowest energy-delay product (default)
        Fastest  // lowest execution time
    };

    inline const char *policy_to_string(OptimizePolicy policy)
    {
        return policy == OptimizePolicy::Fastest ? "fastest" : "edp";
    }

    // Accepts "fastest", "edp" and "best_edp"; anything else yields the default (BestEdp) and sets
    // *recognized to false.
    inline OptimizePolicy string_to_policy(const std::string &str, bool *recognized = nullptr)
    {
        const bool known = (str == "fastest" || str == "edp" || str == "best_edp");
        if (recognized)
            *recognized = known;
        return str == "fastest" ? OptimizePolicy::Fastest : OptimizePolicy::BestEdp;
    }

    // Reads ORBIT_OPTIMIZE_POLICY ("edp" (default), "best_edp" or "fastest"); warns on other values.
    inline OptimizePolicy optimize_policy()
    {
        const char *env = std::getenv("ORBIT_OPTIMIZE_POLICY");
        if (!env || !*env)
            return OptimizePolicy::BestEdp;

        bool recognized = false;
        const OptimizePolicy policy = string_to_policy(env, &recognized);
        if (!recognized)
            std::fprintf(stderr, "ORBIT: unknown ORBIT_OPTIMIZE_POLICY='%s' (expected edp or fastest), using %s\n",
                         env, policy_to_string(policy));
        return policy;
    }
}
