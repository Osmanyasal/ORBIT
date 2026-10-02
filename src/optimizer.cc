#include "interceptor.hh"

namespace orbit {

static void region_begin(Region& region) {
    // apply region optimizations here
    omp_set_num_threads(region.threads);
    omp_set_schedule(region.sched, region.chunk);
}

static void region_end(const Region& region) {
}

}
