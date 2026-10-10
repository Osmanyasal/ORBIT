#include "interceptor.hh"

namespace orbit {

// Runs on the thread initiating the region, right before the real runtime call.
static void region_begin(Region& region) {
    if (detail::ensure_runtime().mode == detail::Mode::Optimize) {
        omp_set_num_threads(region.current.threads);
        omp_set_schedule(region.current.sched, region.current.chunk);
    }
    if (region.current.frequency > 0) {
        for (int socket = 0; socket < OPTKIT_ENV_CPU_NUM_SOCKETS; ++socket) {
            optkit::frequency::cpu::Frequency::set_core_frequency(region.current.frequency, socket);
        }
    }
}

static void region_end(const Region&) {
}

}
