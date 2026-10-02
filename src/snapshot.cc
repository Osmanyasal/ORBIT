#include "interceptor.hh"

namespace orbit {

static void region_begin(Region& region) {
	std::fprintf(stderr, "ORBIT: begin %s fn=%p caller=%p threads=%u chunk=%ld\n",
				 region.entry, reinterpret_cast<void*>(region.function), region.caller,
				 region.threads, region.chunk);
}

static void region_end(const Region& region) {
	std::fprintf(stderr, "ORBIT: end %s fn=%p caller=%p\n", region.entry,
				 reinterpret_cast<void*>(region.function), region.caller);
}

}
