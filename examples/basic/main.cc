#include "orbit.hh"

#include <iostream>
#include <vector>

int main() {
    const std::vector<int> values { 8, 13, 21 };
    std::cout << orbit::parallel_sum(values) << '\n';
    return 0;
}