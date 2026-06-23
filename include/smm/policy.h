#ifndef SMM_POLICY_H
#define SMM_POLICY_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

#include "smm/layer.h"

// On-chip memory policies (Section 3.2 of the paper).
//
//   intra-layer : the whole layer fits on-chip; every element transferred once.
//   Policy 1    : ifmap reuse           - all filters resident, ifmap streamed as
//                                          a height-wise sliding window.
//   Policy 2    : filter reuse          - whole ifmap resident, filters streamed
//                                          one by one.
//   Policy 3    : per-channel reuse     - one channel of every filter resident,
//                                          whole ofmap resident.
//   Policy 4    : partial ifmap reuse   - Policy 1 with filters in blocks of n;
//                                          ifmap re-streamed x = ceil(F#/n) times.
//   Policy 5    : partial per-channel   - Policy 3 with filters in blocks of n.
//
// Each policy fixes the tile sizes of the three data types and therefore the
// GLB footprint (Eq. 1 / Eq. 2) and the off-chip access pattern.

namespace smm {

enum class Policy { Intra, P1, P2, P3, P4, P5 };

inline const char* policy_name(Policy p)
{
    switch (p) {
        case Policy::Intra: return "intra-layer";
        case Policy::P1:    return "policy1";
        case Policy::P2:    return "policy2";
        case Policy::P3:    return "policy3";
        case Policy::P4:    return "policy4";
        case Policy::P5:    return "policy5";
    }
    return "?";
}

inline constexpr std::array<Policy, 6> all_policies()
{
    return {Policy::Intra, Policy::P1, Policy::P2, Policy::P3, Policy::P4, Policy::P5};
}

inline bool is_partial(Policy p) { return p == Policy::P4 || p == Policy::P5; }

// -----------------------------------------------------------------------------
// Working-set size (in elements) of a policy for a layer -- the I_Tile + F_Tile
// + O_Tile sum of Section 3.2.  `n` is the filter-block size for Policy 4/5.
// -----------------------------------------------------------------------------
inline i64 memory_elems(Policy p, const Layer& L, i64 n = 0)
{
    const i64 IH = L.IH, IW = L.IW, FH = L.FH, FW = L.FW;
    const i64 CI = L.CI, Fn = L.Fn, OH = L.OH(), OW = L.OW(), CO = L.CO();
    switch (p) {
        case Policy::Intra:
            return L.ifmap_elems() + L.filter_elems() + L.ofmap_elems();
        case Policy::P1:  // all filters + sliding-window ifmap + one ofmap row
            return FH * IW * CI + FH * FW * CI * Fn + OW * CO;
        case Policy::P2:  // whole ifmap + one filter + one ofmap plane
            return IH * IW * CI + FH * FW * CI + OH * OW;
        case Policy::P3:  // one channel of every filter + 1ch ifmap tile + whole ofmap
            return FH * IW + FH * FW * Fn + OH * OW * CO;
        case Policy::P4:  // Policy 1 with n filters at a time
            return FH * IW * CI + FH * FW * CI * n + OW * n;
        case Policy::P5:  // Policy 3 with n filters at a time
            return FH * IW + FH * FW * n + OH * OW * n;
    }
    return 0;
}

// Largest filter-block size n in [1, F#] whose working set fits in `glb_elems`.
// Returns 0 when even a single filter does not fit (policy infeasible).
inline i64 choose_block(Policy p, const Layer& L, i64 glb_elems)
{
    i64 best = 0;
    for (i64 n = 1; n <= L.Fn; ++n) {
        if (memory_elems(p, L, n) <= glb_elems) {
            best = n;
        } else if (n > 1) {
            break;  // working set grows monotonically in n
        }
    }
    return best;
}

// Minimum GLB footprint of a policy for a layer (smallest feasible working set).
// For Policy 4/5 the minimum is one filter (n = 1).
inline i64 min_memory_elems(Policy p, const Layer& L)
{
    return is_partial(p) ? memory_elems(p, L, 1) : memory_elems(p, L, 0);
}

// -----------------------------------------------------------------------------
// Off-chip access volume (in bytes) for a (policy, layer), analytically per the
// paper's Section 3.2 -- NO cache/LRU.  The scratchpad is SW-managed and the
// off-chip volume is closed-form: every data type is transferred exactly once,
// except the partial policies (P4/P5) which re-load the ifmap x = ceil(F#/n)
// times (the paper's only reload term).  `n` is the filter-block size for P4/P5.
//
// Algorithm 1 only ever evaluates policies whose footprint fits the GLB, so the
// "transferred once" assumption holds for the non-partial policies by
// construction (no eviction modeling needed).
// -----------------------------------------------------------------------------
inline std::size_t estimate_accesses(Policy p, const Layer& L, i64 n, i64 bpe)
{
    const i64 reload = is_partial(p) ? (L.Fn + n - 1) / n : 1;     // ceil(F#/n)
    const i64 elems  = L.ifmap_elems() * reload + L.filter_elems() + L.ofmap_elems();
    return static_cast<std::size_t>(elems) * static_cast<std::size_t>(bpe);
}

}  // namespace smm

#endif  // SMM_POLICY_H
