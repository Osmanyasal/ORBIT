#include <omp.h>
#include <cstdio>

int main() {
    int parallel_threads = 0;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        parallel_threads = omp_get_num_threads();
    }
    std::printf("Parallel region: %d threads\n", parallel_threads);

    int loop_threads = 0;
    int sum = 0;
    #pragma omp parallel for num_threads(4) schedule(dynamic, 2) reduction(+:sum)
    for (int index = 0; index < 16; ++index) {
        if (index == 0) {
            loop_threads = omp_get_num_threads();
        }
        sum += index + 1;
    }
    std::printf("Parallel loop: %d threads, sum=%d (expected 136)\n", loop_threads, sum);

    int runtime_threads = 0;
    int runtime_sum = 0;
    omp_sched_t runtime_schedule = omp_sched_static;
    int runtime_chunk = 0;
    #pragma omp parallel for num_threads(4) schedule(runtime) reduction(+:runtime_sum)
    for (int index = 0; index < 16; ++index) {
        if (index == 0) {
            runtime_threads = omp_get_num_threads();
            omp_get_schedule(&runtime_schedule, &runtime_chunk);
        }
        runtime_sum += index + 1;
    }
    const char* schedule_name = "unknown";
    switch (static_cast<unsigned>(runtime_schedule) & 0x7fffffffU) {
    case omp_sched_static: schedule_name = "static"; break;
    case omp_sched_dynamic: schedule_name = "dynamic"; break;
    case omp_sched_guided: schedule_name = "guided"; break;
    case omp_sched_auto: schedule_name = "auto"; break;
    }
    std::printf("Runtime loop: %d threads, schedule=%s, chunk=%d, sum=%d (expected 136)\n",
                runtime_threads, schedule_name, runtime_chunk, runtime_sum);
    return sum == 136 && runtime_sum == 136 ? 0 : 1;
}