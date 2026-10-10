#include <omp.h>
#include <cstdio>
#include <vector>
#include <cstdint>

int main() {
    // =========================================================================
    // 1. Compute-Bound Parallel Region (>2 seconds)
    // Pure floating-point math in CPU registers/L1, near-zero L3 cache misses,
    // maximum GFLOP/s, and high Arithmetic Intensity.
    // =========================================================================
    int parallel_threads = 0;
    double compute_acc = 0.0;
    #pragma omp parallel reduction(+:compute_acc) num_threads(16)
    {
        #pragma omp single
        parallel_threads = omp_get_num_threads();

        const double t_start = omp_get_wtime();
        double a = 0.5 + (omp_get_thread_num() % 16) * 0.01;
        double b = 0.6;
        double c = 0.7;
        double d = 0.8;

        while (omp_get_wtime() - t_start < 2.2) {
            #pragma unroll(8)
            for (int k = 0; k < 10000; ++k) {
                a = a * 0.5 + b * 0.5 + 0.01;
                b = b * 0.5 + c * 0.5 + 0.02;
                c = c * 0.5 + d * 0.5 + 0.03;
                d = d * 0.5 + a * 0.5 + 0.04;
            }
        }
        compute_acc += (a + b + c + d);
    }
    std::printf("Parallel region (compute-bound): %d threads, acc=%.2f\n",
                parallel_threads, compute_acc);

    // =========================================================================
    // 2. Mixed Parallel Region (>2 seconds)
    // Balanced compute and memory: working set partially fits in L2/L3 cache,
    // with multiple arithmetic operations per memory load/store.
    // =========================================================================
    int loop_threads = 0;
    int sum = 0;
    const size_t MIX_SIZE = 4 * 1024 * 1024; // 4M doubles = 32 MB per vector (~64 MB total)
    std::vector<double> mix_a(MIX_SIZE, 1.0);
    std::vector<double> mix_b(MIX_SIZE, 2.0);

    bool mix_done = false;
    const double mix_start = omp_get_wtime();
    #pragma omp parallel reduction(+:sum)
    {
        if (omp_get_thread_num() == 0) {
            loop_threads = omp_get_num_threads();
        }
        while (!mix_done) {
            #pragma omp for
            for (size_t index = 0; index < MIX_SIZE; ++index) {
                double va = mix_a[index];
                double vb = mix_b[index];
                #pragma unroll(4)
                for (int step = 0; step < 8; ++step) {
                    va = va * 0.85 + vb * 0.15 + 0.01;
                    vb = vb * 0.85 + va * 0.15 + 0.02;
                }
                mix_a[index] = va;
            }
            #pragma omp single
            {
                sum += 1;
                if (omp_get_wtime() - mix_start >= 2.2) {
                    mix_done = true;
                }
            }
        }
    }
    std::printf("Parallel loop (mixed compute/memory): %d threads, passes=%d, val=%.2f\n",
                loop_threads, sum, mix_a[0]);

    // =========================================================================
    // 3. Memory-Bound Runtime Scheduled Loop (>2 seconds)
    // Working set significantly exceeds L3 cache (768 MB total >> 96 MB L3),
    // performing a STREAM Triad with very low arithmetic intensity to saturate
    // DRAM bandwidth and induce high L3 MPKI. Respects schedule(runtime).
    // =========================================================================
    int runtime_threads = 0;
    int runtime_sum = 0;
    omp_sched_t runtime_schedule = omp_sched_static;
    int runtime_chunk = 0;

    const size_t MEM_SIZE = 32 * 1024 * 1024; // 32M doubles = 256 MB per vector (768 MB total)
    std::vector<double> mem_a(MEM_SIZE, 1.0);
    std::vector<double> mem_b(MEM_SIZE, 2.0);
    std::vector<double> mem_c(MEM_SIZE, 3.0);

    bool mem_done = false;
    const double mem_start = omp_get_wtime();
    #pragma omp parallel reduction(+:runtime_sum)
    {
        if (omp_get_thread_num() == 0) {
            runtime_threads = omp_get_num_threads();
            omp_get_schedule(&runtime_schedule, &runtime_chunk);
        }
        while (!mem_done) {
            #pragma omp for schedule(runtime)
            for (size_t index = 0; index < MEM_SIZE; ++index) {
                mem_a[index] = mem_b[index] + 1.5 * mem_c[index];
            }
            #pragma omp single
            {
                runtime_sum += 1;
                if (mem_a[0] > 0.0 && omp_get_wtime() - mem_start >= 2.2) {
                    mem_done = true;
                }
            }
        }
    }
    const char* schedule_name = "unknown";
    switch (static_cast<unsigned>(runtime_schedule) & 0x7fffffffU) {
    case omp_sched_static: schedule_name = "static"; break;
    case omp_sched_dynamic: schedule_name = "dynamic"; break;
    case omp_sched_guided: schedule_name = "guided"; break;
    case omp_sched_auto: schedule_name = "auto"; break;
    }
    std::printf("Runtime loop (memory-bound): %d threads, schedule=%s, chunk=%d, passes=%d, val=%.2f\n",
                runtime_threads, schedule_name, runtime_chunk, runtime_sum, mem_a[0]);

    return (sum > 0 && runtime_sum > 0) ? 0 : 1;
}