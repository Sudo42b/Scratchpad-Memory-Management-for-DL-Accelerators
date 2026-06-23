#ifndef SMM_SCHEDULE_H
#define SMM_SCHEDULE_H

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "smm/layer.h"
#include "smm/manager.h"  // HwParams, layer_macs
#include "smm/policy.h"

// Explicit scratchpad-management mechanism (the "how", on top of the analytical
// "what" in policy.h / manager.h).
//
// Where policy.h returns closed-form footprint/accesses, this header actually
//   (1) PARTITIONS the GLB into ifmap / filter / ofmap regions at byte offsets
//       (Fig. 1 of the paper; x2 per region when double-buffering, Eq. 2), and
//   (2) RUNS the policy's loop nest as a stream of mvin / compute / mvout events,
//       accumulating the off-chip transfer volume, the peak occupancy, and a
//       software-pipelined latency (loads of the next tile overlap the current
//       compute when prefetching).
//
// The simulated off-chip volume equals `estimate_accesses` by construction, so
// the schedule is a faithful refinement of the analytical model, not a second
// guess at the numbers.

namespace smm {

// I_Tile / F_Tile / O_Tile of a policy, in elements (the three managed regions).
struct TileSizes {
    i64 ifmap, filter, ofmap;
};

inline TileSizes tile_sizes(Policy p, const Layer& L, i64 n)
{
    const i64 IH = L.IH, IW = L.IW, FH = L.FH, FW = L.FW;
    const i64 CI = L.CI, Fn = L.Fn, OH = L.OH(), OW = L.OW(), CO = L.CO();
    switch (p) {
        case Policy::Intra: return {L.ifmap_elems(), L.filter_elems(), L.ofmap_elems()};
        case Policy::P1:    return {FH * IW * CI, FH * FW * CI * Fn, OW * CO};
        case Policy::P2:    return {IH * IW * CI, FH * FW * CI, OH * OW};
        case Policy::P3:    return {FH * IW, FH * FW * Fn, OH * OW * CO};
        case Policy::P4:    return {FH * IW * CI, FH * FW * CI * n, OW * n};
        case Policy::P5:    return {FH * IW, FH * FW * n, OH * OW * n};
    }
    return {0, 0, 0};
}

// A region physically allocated in the GLB.
struct Region {
    std::string name;
    std::size_t offset;
    std::size_t bytes;
};

struct GlbLayout {
    std::size_t         capacity = 0;
    std::vector<Region> regions;
    std::size_t         used() const
    {
        std::size_t u = 0;
        for (const auto& r : regions) u += r.bytes;
        return u;
    }
    bool fits() const { return used() <= capacity; }
};

// Lay out the three (optionally double-buffered) regions with a bump allocator.
inline GlbLayout build_layout(Policy p, const Layer& L, std::size_t glb_bytes, const HwParams& hw, bool prefetch, i64 n)
{
    const TileSizes t    = tile_sizes(p, L, n);
    const int       copies = prefetch ? 2 : 1;  // Eq. 2: double buffer
    GlbLayout       lay;
    lay.capacity = glb_bytes;
    std::size_t off = 0;
    auto add = [&](const std::string& name, i64 elems) {
        for (int c = 0; c < copies; ++c) {
            const std::size_t b = static_cast<std::size_t>(elems) * hw.bytes_per_elem;
            lay.regions.push_back({name + (copies == 2 ? (c == 0 ? " (buf A)" : " (buf B)") : ""), off, b});
            off += b;
        }
    };
    add("ifmap", t.ifmap);
    add("filter", t.filter);
    add("ofmap", t.ofmap);
    return lay;
}

struct ScheduleResult {
    GlbLayout   layout;
    std::size_t mvin_ifmap  = 0;  // off-chip bytes, ifmap loads
    std::size_t mvin_filter = 0;  // off-chip bytes, filter loads
    std::size_t mvout_ofmap = 0;  // off-chip bytes, ofmap stores
    long        n_mvin = 0, n_mvout = 0, n_compute = 0;
    std::size_t peak_occupancy = 0;
    double      compute_cycles = 0, transfer_cycles = 0;
    double      latency_serial = 0;     // no overlap: Σ(load+compute+store)
    double      latency_pipelined = 0;  // double-buffered: max(dma,pe) + fill

    std::size_t off_chip() const { return mvin_ifmap + mvin_filter + mvout_ofmap; }
};

// Simulate one layer under a policy: partition the GLB, then stream the tiles.
inline ScheduleResult simulate_layer(Policy p, const Layer& L, std::size_t glb_bytes, const HwParams& hw, bool prefetch, i64 n)
{
    ScheduleResult r;
    r.layout         = build_layout(p, L, glb_bytes, hw, prefetch, n);
    r.peak_occupancy = r.layout.used();

    const i64    bpe = hw.bytes_per_elem;
    const i64    IH = L.IH, IW = L.IW, FH = L.FH, FW = L.FW, CI = L.CI, Fn = L.Fn;
    const i64    OH = L.OH(), OW = L.OW(), CO = L.CO();
    double       dma_busy = 0, pe_busy = 0, first_load_cycles = -1;

    // One pipeline step: bring tiles on-chip, compute, write a result tile.
    auto emit = [&](i64 ifmap_e, i64 filter_e, double macs, i64 ofmap_e) {
        const std::size_t in_b  = static_cast<std::size_t>((ifmap_e + filter_e)) * bpe;
        const std::size_t out_b = static_cast<std::size_t>(ofmap_e) * bpe;
        r.mvin_ifmap += static_cast<std::size_t>(ifmap_e) * bpe;
        r.mvin_filter += static_cast<std::size_t>(filter_e) * bpe;
        r.mvout_ofmap += out_b;
        if (in_b) ++r.n_mvin;
        if (out_b) ++r.n_mvout;
        if (macs > 0) ++r.n_compute;
        const double dma = static_cast<double>(in_b + out_b) / hw.bw_bytes_per_cycle;
        const double pe  = macs / hw.mac_per_cycle;
        dma_busy += dma;
        pe_busy += pe;
        if (first_load_cycles < 0 && in_b) first_load_cycles = static_cast<double>(in_b) / hw.bw_bytes_per_cycle;
        r.latency_serial += dma + pe;  // fully serialized lower path
    };

    switch (p) {
        case Policy::Intra: {
            emit(L.ifmap_elems(), L.filter_elems(), layer_macs(L), L.ofmap_elems());
            break;
        }
        case Policy::P1: {  // filters resident; stream ifmap row-window; ofmap row out
            emit(0, L.filter_elems(), 0, 0);  // prologue: load all filters
            i64 loaded = 0;                    // spread the IH ifmap rows across the OH steps
            for (i64 orow = 0; orow < OH; ++orow) {
                const i64 target = IH * (orow + 1) / OH;
                emit((target - loaded) * IW * CI, 0, static_cast<double>(OW) * CO * FH * FW * CI, OW * CO);
                loaded = target;
            }
            break;
        }
        case Policy::P2: {  // whole ifmap resident; stream filters one by one
            emit(L.ifmap_elems(), 0, 0, 0);  // prologue: load whole ifmap
            for (i64 f = 0; f < Fn; ++f)
                emit(0, FH * FW * CI, static_cast<double>(OH) * OW * FH * FW * CI, OH * OW);
            break;
        }
        case Policy::P3: {  // one channel of every filter resident; ofmap resident, out once
            for (i64 c = 0; c < CI; ++c)
                emit(IH * IW, FH * FW * Fn, static_cast<double>(OH) * OW * CO * FH * FW, 0);
            emit(0, 0, 0, L.ofmap_elems());  // epilogue: write whole ofmap once
            break;
        }
        case Policy::P4: {  // P1 with n filters/block; ifmap re-streamed per block
            const i64 nb = (Fn + n - 1) / n;
            for (i64 b = 0; b < nb; ++b) {
                const i64 nf = std::min<i64>(n, Fn - b * n);
                emit(0, FH * FW * CI * nf, 0, 0);  // load this filter block
                i64 loaded = 0;                    // ifmap re-streamed once per block (IH rows)
                for (i64 orow = 0; orow < OH; ++orow) {
                    const i64 target = IH * (orow + 1) / OH;
                    emit((target - loaded) * IW * CI, 0, static_cast<double>(OW) * nf * FH * FW * CI, OW * nf);
                    loaded = target;
                }
            }
            break;
        }
        case Policy::P5: {  // P3 with n filters/block; per-channel ifmap re-streamed per block
            const i64 nb = (Fn + n - 1) / n;
            for (i64 b = 0; b < nb; ++b) {
                const i64 nf = std::min<i64>(n, Fn - b * n);
                for (i64 c = 0; c < CI; ++c)
                    emit(IH * IW, FH * FW * nf, static_cast<double>(OH) * OW * nf * FH * FW, 0);
                emit(0, 0, 0, static_cast<i64>(OH) * OW * nf);  // write this block's ofmap channels
            }
            break;
        }
    }

    r.compute_cycles  = pe_busy;
    r.transfer_cycles = dma_busy;
    // Double-buffered: DMA and PE run on separate engines; the first load can't
    // be hidden (pipeline fill).  Otherwise everything is serialized.
    const double fill        = first_load_cycles > 0 ? first_load_cycles : 0;
    r.latency_pipelined      = std::max(dma_busy, pe_busy) + fill;
    return r;
}

}  // namespace smm

#endif  // SMM_SCHEDULE_H
