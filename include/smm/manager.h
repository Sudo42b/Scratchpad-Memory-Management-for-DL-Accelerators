#ifndef SMM_MANAGER_H
#define SMM_MANAGER_H

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "smm/layer.h"
#include "smm/policy.h"

// Memory management technique (Section 3.3, Algorithm 1).
//
// For each layer and each candidate policy we estimate the GLB footprint, the
// off-chip access volume and the latency.  Among the policies whose footprint
// fits the GLB, the analyser keeps the one that best matches the optimization
// objective (fewest off-chip accesses, or lowest latency).  Doing this per
// layer yields a *heterogeneous* execution plan; forcing one policy for the
// whole network yields a *homogeneous* plan.

namespace smm {

// Accelerator specifications (Section 4 experimental setup): 16x16 PE array,
// output-stationary, 8-bit data, off-chip bandwidth of 16 elements/cycle.
struct HwParams {
    i64    bytes_per_elem     = 1;    // 8-bit data width
    double mac_per_cycle      = 256;  // 16x16 = 256 MACs/cycle (OPs/cycle = 512)
    double bw_bytes_per_cycle = 16;   // 16 elements/cycle at 8-bit
};

enum class Objective { Accesses, Latency };

// Number of multiply-accumulate operations of a layer.
inline double layer_macs(const Layer& L)
{
    return static_cast<double>(L.OH()) * L.OW() * L.CO() * L.FH * L.FW * L.CI;
}

struct LayerPlan {
    Policy      policy   = Policy::P2;
    i64         n        = 0;      // filter-block size (Policy 4/5)
    bool        prefetch = false;
    std::size_t memory   = 0;      // GLB footprint in bytes (Eq. 1 / Eq. 2)
    std::size_t accesses = 0;      // off-chip transfer volume in bytes
    double      latency  = 0;      // latency in cycles
    bool        feasible = false;
};

// Evaluate one (policy, prefetch) candidate for a layer on a given GLB.
inline LayerPlan evaluate(Policy p, bool prefetch, const Layer& L, std::size_t glb_bytes, const HwParams& hw)
{
    LayerPlan plan;
    plan.policy   = p;
    plan.prefetch = prefetch;

    // Double buffering (Eq. 2) reserves half of the GLB for the prefetch copy,
    // so only half is available for the active working set.
    const i64 mult     = prefetch ? 2 : 1;
    const i64 glb_elems_eff = static_cast<i64>(glb_bytes / hw.bytes_per_elem) / mult;

    i64 n = 0;
    if (is_partial(p)) {
        n = choose_block(p, L, glb_elems_eff);
        if (n == 0) return plan;  // not even one filter fits -> infeasible
    } else if (min_memory_elems(p, L) > glb_elems_eff) {
        return plan;              // working set does not fit -> infeasible
    }
    plan.n = n;

    plan.memory = static_cast<std::size_t>(memory_elems(p, L, n)) * hw.bytes_per_elem * mult;

    // Off-chip accesses, analytically per Section 3.2 (no cache/LRU).
    plan.accesses = estimate_accesses(p, L, n, hw.bytes_per_elem);

    const double compute_cycles  = layer_macs(L) / hw.mac_per_cycle;
    const double transfer_cycles = static_cast<double>(plan.accesses) / hw.bw_bytes_per_cycle;
    // Without prefetching transfers and compute are serialized; prefetching
    // overlaps them, hiding the smaller of the two behind the larger.
    plan.latency = prefetch ? std::max(compute_cycles, transfer_cycles) : (compute_cycles + transfer_cycles);

    plan.feasible = true;
    return plan;
}

inline double metric(const LayerPlan& plan, Objective obj)
{
    return obj == Objective::Accesses ? static_cast<double>(plan.accesses) : plan.latency;
}

// Algorithm 1: pick the best feasible policy for a single layer.
// `allow_prefetch` enables the "+p" double-buffered candidates.
inline LayerPlan best_for_layer(const Layer& L, std::size_t glb_bytes, const HwParams& hw, Objective obj, bool allow_prefetch)
{
    LayerPlan best;
    double    best_metric = std::numeric_limits<double>::infinity();
    for (Policy p : all_policies()) {
        for (bool pf : {false, true}) {
            if (pf && !allow_prefetch) continue;
            LayerPlan cand = evaluate(p, pf, L, glb_bytes, hw);
            if (!cand.feasible) continue;
            const double m = metric(cand, obj);
            if (m < best_metric) {
                best_metric = m;
                best        = cand;
            }
        }
    }
    return best;
}

struct SchemeResult {
    std::vector<LayerPlan> plans;       // per-layer decision
    std::size_t            accesses = 0;
    double                 latency  = 0;
    bool                   feasible = true;
};

// Heterogeneous management scheme: best policy per layer (Algorithm 1).
inline SchemeResult heterogeneous(const Model& m, std::size_t glb_bytes, const HwParams& hw, Objective obj, bool allow_prefetch)
{
    SchemeResult r;
    for (const Layer& L : m.layers) {
        LayerPlan plan = best_for_layer(L, glb_bytes, hw, obj, allow_prefetch);
        if (!plan.feasible) r.feasible = false;
        r.accesses += plan.accesses;
        r.latency  += plan.latency;
        r.plans.push_back(plan);
    }
    return r;
}

// Homogeneous management scheme: the single policy (optionally prefetched) that
// is feasible for every layer and minimizes the total objective.
inline SchemeResult homogeneous(const Model& m, std::size_t glb_bytes, const HwParams& hw, Objective obj, bool allow_prefetch)
{
    SchemeResult best;
    double       best_total = std::numeric_limits<double>::infinity();
    bool         found      = false;

    for (Policy p : all_policies()) {
        for (bool pf : {false, true}) {
            if (pf && !allow_prefetch) continue;
            SchemeResult r;
            bool         ok = true;
            for (const Layer& L : m.layers) {
                LayerPlan plan = evaluate(p, pf, L, glb_bytes, hw);
                if (!plan.feasible) { ok = false; break; }
                r.accesses += plan.accesses;
                r.latency  += plan.latency;
                r.plans.push_back(plan);
            }
            if (!ok) continue;
            const double total = (obj == Objective::Accesses) ? static_cast<double>(r.accesses) : r.latency;
            if (total < best_total) {
                best_total = total;
                best       = r;
                found      = true;
            }
        }
    }
    best.feasible = found;
    return best;
}

}  // namespace smm

#endif  // SMM_MANAGER_H
