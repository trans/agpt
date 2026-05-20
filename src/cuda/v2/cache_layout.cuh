#ifndef AGPT_V2_CACHE_LAYOUT_CUH
#define AGPT_V2_CACHE_LAYOUT_CUH

#include "types.cuh"

namespace agpt_v2 {

enum class KCoordinateSpace {
    PreRope,
    PostRope,
};

struct CacheLayout {
    RuntimeShape shape;
    KCoordinateSpace k_space = KCoordinateSpace::PostRope;
    bool v_is_rope_free = true;
    bool compact_slot_indexed = true;
};

static inline CacheLayout make_cache_layout(const RuntimeShape& shape) {
    CacheLayout layout;
    layout.shape = shape;
    return layout;
}

}  // namespace agpt_v2

#endif
