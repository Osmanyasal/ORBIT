#include "interceptor.hh"

namespace orbit {

// Runs on the thread initiating the region, right before the real runtime call.
static void region_begin(Region& region) {
    if (detail::ensure_runtime().mode == detail::Mode::Optimize) {
        omp_set_num_threads(region.threads);
        omp_set_schedule(region.sched, region.chunk);
    }
    if (region.frequency > 0) {
        for (int socket = 0; socket < OPTKIT_ENV_CPU_NUM_SOCKETS; ++socket) {
            optkit::frequency::cpu::Frequency::set_core_frequency(region.frequency, socket);
        }
    }
}

static void region_end(const Region&) {
}

}
