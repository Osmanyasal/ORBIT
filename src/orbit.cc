#include "interceptor.hh"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace orbit {

namespace detail {

using Clock = std::chrono::steady_clock;
using CpuFrequency = optkit::frequency::cpu::Frequency;

// Writing the CPU frequency touches sysfs for every core, which is slow next to a region of a few
// milliseconds that is called thousands of times. The governor therefore
//  - writes a frequency only when it differs from the one in effect, so a hot loop of the same
//    region sets it once on the first call and never again, and
//  - restores the baseline (the ORBIT_CPU_FREQ value, or the system default) only when the regions
//    stop: no region has been running for a grace period. A watchdog thread detects that, since
//    nothing else happens when a region is simply not entered again.
// A different region with another frequency switches directly, without an intermediate reset.
// OPTKIT restores the original frequency at exit, and the governor stops its watchdog before that.
class FrequencyGovernor {
public:
    static FrequencyGovernor& instance() {
        // ensure_runtime() registers the OPTKIT atexit handler first, so this static is destroyed first.
        static FrequencyGovernor governor{ensure_runtime().frequency};
        return governor;
    }

    FrequencyGovernor(const FrequencyGovernor&) = delete;
    FrequencyGovernor& operator=(const FrequencyGovernor&) = delete;

    ~FrequencyGovernor() {
        this->stop_watchdog();
    }

    // A region starts: apply its frequency unless it is already in effect. 0 means no frequency was
    // configured, which targets the baseline.
    void enter(std::int64_t requested) {
        std::lock_guard<std::mutex> lock(this->mutex_);
        ++this->active_regions_;
        ++this->activity_;

        const std::int64_t target = requested > 0 ? requested : this->baseline_;
        if (target != this->applied_)
            this->apply(target);
        if (this->applied_ != this->baseline_)
            this->start_watchdog();
    }

    // A region ends: only record it. The watchdog restores the baseline if no region follows.
    void leave() {
        bool wake_watchdog;
        {
            std::lock_guard<std::mutex> lock(this->mutex_);
            if (this->active_regions_ > 0)
                --this->active_regions_;
            ++this->activity_;
            this->last_leave_ = Clock::now();
            // A watchdog in its timed wait notices the activity at its deadline by itself; waking it on
            // every region end would cost a system call per call.
            wake_watchdog = this->sleeping_until_idle_ && this->is_held_and_idle();
        }
        if (wake_watchdog)
            this->wakeup_.notify_one();
    }

private:
    // Quiet time after the last region ended before the frequency is restored: at least this long,
    // and never less than a multiple of the measured switch cost.
    static constexpr double min_grace_s = 0.02;
    static constexpr double grace_switch_cost_factor = 10.0;

    explicit FrequencyGovernor(std::int64_t baseline)
        : baseline_{baseline}, applied_{baseline}, active_regions_{0}, activity_{0},
          switch_cost_s_{0.0}, stop_{false}, sleeping_until_idle_{false}, last_leave_{Clock::now()} {}

    // True when a non-baseline frequency is in effect while no region is running. The caller holds the mutex.
    bool is_held_and_idle() const {
        return this->applied_ != this->baseline_ && this->active_regions_ == 0;
    }

    // Applies the frequency to all sockets, 0 meaning the system default, and records how long the
    // switch took. The caller holds the mutex.
    void apply(std::int64_t frequency) {
        const auto start = Clock::now();
        bool all_sockets_ok = true;
        for (int socket = 0; socket < OPTKIT_ENV_CPU_NUM_SOCKETS; ++socket) {
            const bool ok = frequency > 0 ? CpuFrequency::set_core_frequency(frequency, socket)
                                          : CpuFrequency::reset_core_frequency(socket);
            all_sockets_ok = all_sockets_ok && ok;
        }
        if (all_sockets_ok) {
            this->applied_ = frequency;
            this->switch_cost_s_ = std::chrono::duration<double>(Clock::now() - start).count();
        }
    }

    Clock::duration grace_period() const {
        const double seconds = std::max(min_grace_s, grace_switch_cost_factor * this->switch_cost_s_);
        return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    }

    // The caller holds the mutex.
    void start_watchdog() {
        if (!this->watchdog_.joinable())
            this->watchdog_ = std::thread(&FrequencyGovernor::watch, this);
    }

    void stop_watchdog() {
        {
            std::lock_guard<std::mutex> lock(this->mutex_);
            this->stop_ = true;
        }
        this->wakeup_.notify_all();
        if (this->watchdog_.joinable())
            this->watchdog_.join();
    }

    // Restores the baseline once a held frequency has seen no region activity for a grace period.
    void watch() {
        std::unique_lock<std::mutex> lock(this->mutex_);
        while (true) {
            this->sleeping_until_idle_ = true;
            this->wakeup_.wait(lock, [this] { return this->stop_ || this->is_held_and_idle(); });
            this->sleeping_until_idle_ = false;
            if (this->stop_)
                return;
            if (this->wait_for_quiet_period(lock) && this->is_held_and_idle())
                this->apply(this->baseline_);
        }
    }

    // Returns true when no region started or ended during the grace period, false when interrupted
    // by activity or by shutdown.
    bool wait_for_quiet_period(std::unique_lock<std::mutex> &lock) {
        const std::uint64_t activity_before = this->activity_;
        const Clock::time_point deadline = this->last_leave_ + this->grace_period();
        const bool interrupted = this->wakeup_.wait_until(lock, deadline, [this, activity_before] {
            return this->stop_ || this->activity_ != activity_before;
        });
        return !interrupted;
    }

    std::mutex mutex_;
    std::condition_variable wakeup_;
    std::thread watchdog_;
    std::int64_t baseline_;  // frequency to return to, 0 for the system default
    std::int64_t applied_;   // frequency currently in effect, 0 for the system default
    int active_regions_;     // regions currently running, nested or on several threads
    std::uint64_t activity_; // incremented on every region entry and exit
    double switch_cost_s_;   // duration of the last frequency switch
    bool stop_;
    bool sleeping_until_idle_; // the watchdog waits for a held frequency with no running region
    Clock::time_point last_leave_;
};

inline void apply_openmp_settings(const RegionConfig &config) {
    omp_set_num_threads(config.threads);
    omp_set_schedule(config.sched, config.chunk);
}

inline void reset_openmp_settings() {
    omp_set_num_threads(omp_get_max_threads());
    omp_set_schedule(omp_sched_static, 0);
}

}  // namespace detail

// Runs on the thread initiating the region, right before the real runtime call.
static void region_begin(Region& region) {
    if (detail::ensure_runtime().mode == detail::Mode::Optimize)
        detail::apply_openmp_settings(region.current);
    detail::FrequencyGovernor::instance().enter(region.current.frequency);
}

static void region_end(const Region& /*region*/) {
    // Only Optimize mode changes the OpenMP settings, so only there is there anything to undo.
    if (detail::ensure_runtime().mode == detail::Mode::Optimize)
        detail::reset_openmp_settings();
    detail::FrequencyGovernor::instance().leave();
}
}
