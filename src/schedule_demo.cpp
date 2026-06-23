// schedule_demo.cpp — the explicit scratchpad-management mechanism in action.
//
// For each ResNet18 layer it (1) runs Algorithm 1 to pick the per-layer policy,
// (2) physically PARTITIONS the GLB into ifmap/filter/ofmap regions, and
// (3) STREAMS the policy's tiles as mvin/compute/mvout events, reporting the
// off-chip volume, peak occupancy and software-pipelined latency it observed.
// The simulated off-chip volume is checked against the analytical estimate.

#include <cstdio>

#include "smm/manager.h"
#include "smm/models.h"
#include "smm/policy.h"
#include "smm/schedule.h"

using namespace smm;

namespace {

constexpr double KB = 1024.0;
constexpr double MB = 1024.0 * 1024.0;

// Detailed view of one layer: the GLB partition map + the schedule it ran.
void show_layer_detail(const Layer& L, const LayerPlan& plan, const HwParams& hw, std::size_t glb_bytes)
{
    const ScheduleResult s = simulate_layer(plan.policy, L, glb_bytes, hw, plan.prefetch, plan.n);

    std::printf("================ %s : %s%s ================\n", L.name.c_str(), policy_name(plan.policy), plan.prefetch ? " +prefetch" : "");
    std::printf("layer  : %lldx%lld x%lld ifmap, %lldx%lld x%lld x%lld filters, stride %lld pad %lld\n",
                (long long) L.IH, (long long) L.IW, (long long) L.CI, (long long) L.FH, (long long) L.FW, (long long) L.CI, (long long) L.Fn,
                (long long) L.S, (long long) L.P);
    if (plan.n) std::printf("filter block n = %lld  (ifmap re-streamed %lld times)\n", (long long) plan.n, (long long) ((L.Fn + plan.n - 1) / plan.n));

    std::printf("\n-- GLB partition map (capacity %.1f kB) --\n", glb_bytes / KB);
    for (const Region& r : s.layout.regions)
        std::printf("  [%6.2f .. %6.2f] kB  %s\n", r.offset / KB, (r.offset + r.bytes) / KB, r.name.c_str());
    std::printf("  used %.2f / %.1f kB  (%.0f%% utilized)\n", s.layout.used() / KB, glb_bytes / KB, 100.0 * s.layout.used() / glb_bytes);

    std::printf("\n-- schedule (tiles streamed through the partition) --\n");
    std::printf("  mvin  : ifmap %.1f kB + filter %.1f kB   (%ld load events)\n", s.mvin_ifmap / KB, s.mvin_filter / KB, s.n_mvin);
    std::printf("  mvout : ofmap %.1f kB                    (%ld store events)\n", s.mvout_ofmap / KB, s.n_mvout);
    std::printf("  compute events : %ld\n", s.n_compute);
    std::printf("  off-chip total : %.1f kB   (analytical %.1f kB -> %s)\n", s.off_chip() / KB, plan.accesses / KB,
                s.off_chip() == plan.accesses ? "MATCH" : "MISMATCH");
    std::printf("  cycles : compute %.0f, transfer %.0f\n", s.compute_cycles, s.transfer_cycles);
    std::printf("  latency: serial %.0f -> pipelined %.0f  (%.0f%% hidden by double buffering)\n\n", s.latency_serial, s.latency_pipelined,
                100.0 * (1.0 - s.latency_pipelined / s.latency_serial));
}

}  // namespace

int main()
{
    HwParams hw;
    Model    m   = resnet18();
    const std::size_t glb = 64 * 1024;

    std::printf("Explicit scratchpad management — ResNet18 @ %zu kB GLB\n", glb / 1024);
    std::printf("(Algorithm 1 picks the policy; the GLB is partitioned and the tiles are streamed)\n\n");

    // Pick two layers to show in full detail (one filter-heavy, one ifmap-heavy).
    SchemeResult het = heterogeneous(m, glb, hw, Objective::Accesses, /*prefetch=*/true);
    for (std::size_t i = 0; i < m.layers.size(); ++i)
        if (m.layers[i].name == "conv1" || m.layers[i].name == "conv5_1") show_layer_detail(m.layers[i], het.plans[i], hw, glb);

    // Same layer with double buffering (Eq. 2): the GLB is split into A/B copies
    // so the next tile loads while the current one computes -> latency is hidden.
    for (const Layer& L : m.layers)
        if (L.name == "conv5_1") {
            const LayerPlan pf = evaluate(Policy::P2, /*prefetch=*/true, L, glb, hw);
            std::printf(">>> the same layer, now DOUBLE-BUFFERED (prefetch) <<<\n");
            show_layer_detail(L, pf, hw, glb);
        }

    // Per-layer summary across the whole network.
    std::printf("================ per-layer management summary ================\n");
    std::printf("%-10s %-9s %3s %8s %8s %8s %9s %10s %6s\n", "layer", "policy", "pf", "ifmap", "filter", "ofmap", "occ(kB)", "offchip", "ok");
    std::size_t tot_off = 0;
    double      tot_lat = 0;
    for (std::size_t i = 0; i < m.layers.size(); ++i) {
        const Layer&    L = m.layers[i];
        const LayerPlan p = het.plans[i];
        const ScheduleResult s = simulate_layer(p.policy, L, glb, hw, p.prefetch, p.n);
        const TileSizes      t = tile_sizes(p.policy, L, p.n);
        std::printf("%-10s %-9s %3s %8.1f %8.1f %8.1f %9.1f %9.1f %6s\n", L.name.c_str(), policy_name(p.policy), p.prefetch ? "+p" : "-",
                    static_cast<double>(t.ifmap) * hw.bytes_per_elem / KB, static_cast<double>(t.filter) * hw.bytes_per_elem / KB,
                    static_cast<double>(t.ofmap) * hw.bytes_per_elem / KB, s.peak_occupancy / KB, s.off_chip() / KB,
                    s.off_chip() == p.accesses ? "ok" : "BAD");
        tot_off += s.off_chip();
        tot_lat += s.latency_pipelined;
    }
    std::printf("\nTOTAL off-chip %.3f MB   pipelined latency %.3f Mcyc\n", tot_off / MB, tot_lat / 1e6);
    std::printf("(off-chip per layer equals the analytical estimate_accesses -> schedule is faithful)\n");
    return 0;
}
