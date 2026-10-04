#define ORBIT_SNAPSHOT 1

#include "interceptor.hh"

namespace orbit {

static void region_begin(Region& region) {
	// std::fprintf(stderr, "ORBIT: begin name=%s %s fn=%p caller=%p threads=%d sched=%s chunk=%ld\n",
	// 			 region.name.c_str(), region.entry, reinterpret_cast<void*>(region.function), region.caller,
	// 			 region.threads, sched_to_string(region.sched), region.chunk);
	
}

static void region_end(const Region& region) {
	// std::fprintf(stderr, "ORBIT: end name=%s %s fn=%p caller=%p\n", region.name.c_str(), region.entry,
	// 			 reinterpret_cast<void*>(region.function), region.caller);
}

}

