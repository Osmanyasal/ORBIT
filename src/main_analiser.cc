#include <iostream>
#include <vector>
#include "optkit.hh"

int main() {
    OPTKIT_INIT();
    OPTKIT_INFO("Starting main function");
    OPTKIT_CPU_EVENTS("main",optkit::metrics::performance::cpu_metrics::ai());
    return 0;
}