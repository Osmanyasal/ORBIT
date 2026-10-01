#include "orbit.hh"

#include <vector>

int main() {
    const std::vector<int> values { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    return orbit::parallel_sum(values) == 55 ? 0 : 1;
}