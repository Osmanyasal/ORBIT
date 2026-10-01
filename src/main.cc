#include "orbit.hh"

#include <iostream>
#include <vector>

int main() {
    const std::vector<int> values { 1, 2, 3, 4 };
    std::cout << "ORBIT parallel sum: " << orbit::parallel_sum(values) << '\n';
    return 0;
}