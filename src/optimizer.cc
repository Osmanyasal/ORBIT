#define ORBIT_OPTIMIZER 1

#include "interceptor.hh"

namespace orbit {

static void region_begin(Region& region) {
    // apply region optimizations here
    omp_set_num_threads(region.threads);
    omp_set_schedule(region.sched, region.chunk);
    if (region.frequency > 0) {
        for (int socket = 0; socket < OPTKIT_ENV_CPU_NUM_SOCKETS; ++socket) {
            optkit::frequency::cpu::Frequency::set_core_frequency(region.frequency, socket);
        }
    }
}

static void region_end(const Region& region) {
}

}
