#include "orbit.hh"

namespace orbit {

long parallel_sum(const std::vector<int>& values) {
    long total = 0;

#pragma omp parallel for reduction(+ : total)
    for (std::size_t index = 0; index < values.size(); ++index) {
        total += values[index];
    }

    return total;
}

}