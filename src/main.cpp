#include <cstdio>
#include <string>
#include <vector>

#include "smm/layer.h"
#include "smm/manager.h"
#include "smm/models.h"
#include "smm/policy.h"

using namespace smm;

namespace {

constexpr double KB = 1024.0;
constexpr double MB = 1024.0 * 1024.0;

// Maximum working-set footprint of a policy across all layers, in kB (Table 3).
double max_footprint_kb(const Model& m, Policy p, i64 bpe)
{
    i64 mx = 0;
    for (const Layer& L : m.layers) {
        const i64 mem = is_partial(p) ? memory_elems(p, L, 1) : memory_elems(p, L, 0);
        if (mem > mx) mx = mem;
    }
    return static_cast<double>(mx) * bpe / KB;
}

void validate_table3(const Model& m, const HwParams& hw)
{
    std::printf("================ Table 3 validation : %s ================\n", m.name.c_str());
    std::printf("Maximum memory requirement per policy (kB), each element transferred once\n");
    std::printf("  intra-layer : %8.1f   (paper 2353.0)\n", max_footprint_kb(m, Policy::Intra, hw.bytes_per_elem));
    std::printf("  Policy 1    : %8.1f   (all-filters resident -> dominated by full filters)\n", max_footprint_kb(m, Policy::P1, hw.bytes_per_elem));
    std::printf("  Policy 2    : %8.1f   (paper  199.7)\n", max_footprint_kb(m, Policy::P2, hw.bytes_per_elem));
    std::printf("  Policy 3    : %8.1f   (whole-ofmap resident -> dominated by full ofmap)\n", max_footprint_kb(m, Policy::P3, hw.bytes_per_elem));
    std::printf("\n");
    std::printf("Note: the textual policy definitions (Sec 3.2) give Policy1 = full-filter\n");
    std::printf("      footprint (~2318) and Policy3 = full-ofmap footprint (~788.6).  Table 3\n");
    std::printf("      prints these two columns the other way around (P1=788.6, P3=2318): the\n");
    std::printf("      value *set* matches exactly, the P1/P3 column order is a paper typo.\n\n");
}

void print_layer_breakdown(const Model& m, std::size_t glb_bytes, const HwParams& hw)
{
    std::printf("---------------- Heterogeneous per-layer policy @ %zukB (accesses) ----------------\n", (size_t)(glb_bytes / 1024));
    SchemeResult het = heterogeneous(m, glb_bytes, hw, Objective::Accesses, /*prefetch=*/true);
    std::printf("%-10s %-11s %4s %3s %10s %12s\n", "layer", "policy", "n", "pf", "mem(kB)", "acc(kB)");
    for (std::size_t i = 0; i < m.layers.size(); ++i) {
        const LayerPlan& p = het.plans[i];
        std::printf("%-10s %-11s %4lld %3s %10.1f %12.1f\n",
                    m.layers[i].name.c_str(), policy_name(p.policy), (long long) p.n,
                    p.prefetch ? "+p" : "-", p.memory / KB, p.accesses / KB);
    }
    std::printf("TOTAL off-chip accesses: %.3f MB\n\n", het.accesses / MB);
}

void sweep_accesses(const Model& m, const HwParams& hw)
{
    const std::vector<std::size_t> sizes_kb = {64, 128, 256, 512, 1024};
    std::printf("================ Off-chip accesses vs GLB size : %s ================\n", m.name.c_str());
    std::printf("(objective: minimize accesses; values in MB)\n");
    std::printf("%-10s %14s %14s %12s\n", "GLB", "Homogeneous", "Heterogeneous", "Het benefit");
    for (std::size_t kb : sizes_kb) {
        const std::size_t glb = kb * 1024;
        SchemeResult hom = homogeneous(m, glb, hw, Objective::Accesses, /*prefetch=*/true);
        SchemeResult het = heterogeneous(m, glb, hw, Objective::Accesses, /*prefetch=*/true);
        const double hom_mb = hom.feasible ? hom.accesses / MB : -1.0;
        const double het_mb = het.accesses / MB;
        const double benefit = (hom.feasible && hom.accesses > 0) ? 100.0 * (1.0 - het_mb / hom_mb) : 0.0;
        if (hom.feasible)
            std::printf("%-8zukB %14.2f %14.2f %11.1f%%\n", kb, hom_mb, het_mb, benefit);
        else
            std::printf("%-8zukB %14s %14.2f %12s\n", kb, "infeasible", het_mb, "-");
    }
    std::printf("\n");
}

void sweep_latency(const Model& m, const HwParams& hw)
{
    const std::vector<std::size_t> sizes_kb = {64, 128, 256, 512, 1024};
    std::printf("================ Latency vs GLB size : %s ================\n", m.name.c_str());
    std::printf("(Het_a: optimized for accesses | Het_l: optimized for latency; million cycles)\n");
    std::printf("%-10s %14s %14s\n", "GLB", "Het_a", "Het_l");
    for (std::size_t kb : sizes_kb) {
        const std::size_t glb = kb * 1024;
        SchemeResult het_a = heterogeneous(m, glb, hw, Objective::Accesses, true);
        SchemeResult het_l = heterogeneous(m, glb, hw, Objective::Latency, true);
        std::printf("%-8zukB %14.3f %14.3f\n", kb, het_a.latency / 1e6, het_l.latency / 1e6);
    }
    std::printf("\n");
}

}  // namespace

int main()
{
    HwParams hw;  // 8-bit data, 16x16 PEs, 16 B/cycle off-chip bandwidth
    Model    m = resnet18();

    std::printf("Scratchpad Memory Management for Deep Learning Accelerators\n");
    std::printf("Zouzoula et al., ICPP '24 -- analytical model (Sec 3.2 accesses + Sec 3.3 Algorithm 1, no LRU)\n");
    std::printf("Model: %s  (%zu weight layers)\n\n", m.name.c_str(), m.layers.size());

    validate_table3(m, hw);
    sweep_accesses(m, hw);
    print_layer_breakdown(m, 64 * 1024, hw);
    sweep_latency(m, hw);

    return 0;
}
