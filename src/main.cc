#include "orbit.hh"

#include <iostream>
#include <vector>
#include "optkit.hh"

int main() {
    OPTKIT_INIT();

    OPTKIT_CPU_EVENTS("main",optkit::metrics::performance::cpu_metrics::ai());
    const std::vector<int> values { 1, 2, 3, 4 };
    std::cout << "ORBIT parallel sum: " << orbit::parallel_sum(values) << '\n';
    return 0;
}