# Verification: the SMM model vs the SMM paper (ICPP '24)

Paper: S. Zouzoula, M. A. Maleki, M. W. Azhar, P. Trancoso, **"Scratchpad Memory
Management for Deep Learning Accelerators"**, ICPP '24, pp. 629–639,
doi:10.1145/3673038.3673115 (`paper/542648_Fulltext.pdf`).

**Verdict: a faithful, correct *analytical* model of the paper's memory-management
technique.** The five on-chip policies + intra-layer, the GLB constraint
(Eq. 1 / Eq. 2), Algorithm 1, and the ResNet18 hardware setup (§4) are all
implemented exactly as the paper specifies. It builds clean (`-Wall -Wextra`)
and reproduces the Table 3 value set and the Fig. 5 / Fig. 8 trends for ResNet18.

> The GLB (§2.1) is a **software-managed scratchpad**, modeled exactly as the
> paper specifies — **no LRU / no cache**. The off-chip access volume is the
> closed-form estimate of §3.2 (`estimate_accesses` in `policy.h`): every data
> type is transferred once, and the partial policies P4/P5 add the only reload
> term the paper defines, `ifmap × ⌈F#/n⌉`. Algorithm 1 (§3.3) only ever selects
> policies whose footprint ≤ GLB, so the "transferred once" assumption holds by
> construction. Everything is closed-form arithmetic.
>
> *History:* an earlier revision modeled the GLB with a cachemere LRU cache and
> *measured* off-chip volume by replaying each policy's tile schedule. That was
> removed (2026-06-18) because the paper has no cache/LRU — its scratchpad is
> SW-managed and the volume is analytic. The LRU never changed the *result*
> (Algorithm 1 picks only fitting policies, where LRU = "once" = the formula), so
> the only effect of the switch is on the partial-policy reload of layers whose
> ifmap happened to fit the GLB: the LRU saw a resident hit (reload ≈1) while the
> paper's formula charges `⌈F#/n⌉`. The analytic form is the faithful one.

## Correctly implemented (faithful to the paper)

| Paper element | Section | implementation |
|---|---|---|
| Layer hyperparameters (Table 1) | §2.2 | `layer.h` `Layer{IH,IW,FH,FW,CI,Fn,S,P}`, derived `OH/OW/CO` ✓ |
| GLB constraint Eq. (1): `GLB ≥ I+F+O` | §3.1 | `evaluate`: `mult=1`, footprint ≤ glb ✓ |
| GLB constraint Eq. (2): `GLB ≥ 2I+2F+2O` (double buffer) | §3.1 | `evaluate`: `mult=2` on prefetch → `glb_elems_eff = glb/2` ✓ |
| intra-layer: full ifmap+filter+ofmap | §3.2 | `memory_elems(Intra)` = `ifmap+filter+ofmap` ✓ |
| Policy 1 ifmap reuse: `F_H·I_W·C_I` + full filters + `O_W·C_O` | §3.2 | `FH*IW*CI + FH*FW*CI*Fn + OW*CO` ✓ |
| Policy 2 filter reuse: full ifmap + `F_H·F_W·C_I` + `O_H·O_W` | §3.2 | `IH*IW*CI + FH*FW*CI + OH*OW` ✓ |
| Policy 3 per-channel: `F_H·I_W` + `F_H·F_W·F#` + `O_H·O_W·C_O` | §3.2 | `FH*IW + FH*FW*Fn + OH*OW*CO` ✓ |
| Policy 4 partial ifmap (n filters): `…+ F_H·F_W·C_I·n + O_W·n`, reload `⌈F#/n⌉` | §3.2 | `memory_elems(P4,n)` + `estimate_accesses` ifmap×⌈F#/n⌉ ✓ |
| Policy 5 partial per-channel (n filters): `…+ F_H·F_W·n + O_H·O_W·n`, reload `⌈F#/n⌉` | §3.2 | `memory_elems(P5,n)` + `estimate_accesses` ifmap×⌈F#/n⌉ ✓ |
| off-chip accesses (Sec 3.2): once + P4/P5 reload | §3.2 | `estimate_accesses` = `(ifmap·⌈F#/n⌉ + filter + ofmap)·bpe` (closed-form, no LRU) ✓ |
| Algorithm 1: per-layer best feasible policy | §3.3 | `best_for_layer`: candidates = {Intra,P1..P5}×{no-pf,pf}, keep min(accesses|latency) among feasible ✓ |
| Homogeneous vs Heterogeneous scheme | §3.3 | `homogeneous` (one policy fits all layers) / `heterogeneous` (per-layer) ✓ |
| HW setup: 16×16 PEs, OS, 8-bit, 16 B/cyc off-chip BW | §4 | `HwParams{bytes_per_elem=1, mac_per_cycle=256, bw_bytes_per_cycle=16}` ✓ |
| Latency: prefetch overlaps transfer with compute | §5.2 | `latency = pf ? max(compute,transfer) : compute+transfer` (roofline) ✓ |
| ResNet18 (Table 2, 21 weight layers) | §4 | `models.h resnet18()`: conv1 + 16 block convs + 3 projections + fc ✓ |

## Quantitative regression (build/run baseline, this session)

| metric | this code | paper | note |
|---|---|---|---|
| Table 3 intra footprint | 2353.0 kB | 2353.0 | exact |
| Table 3 P2 (full-ifmap) | 199.6 kB | 199.7 | rounding |
| Table 3 P1 (full-filter, conv5) | 2318.0 kB | 2318 (paper col "P3") | hand-recomputed 2318.0 ✓ |
| Table 3 P3 (full-ofmap, conv1) | 788.6 kB | 788.6 (paper col "P1") | hand-recomputed 788.6 ✓ |
| ResNet18 het @64 kB accesses | 16.07 MB | ≈Fig.5(e) | analytic §3.2 (was 16.19 under old LRU) |
| ResNet18 homo @256 kB accesses | 15.59 MB | ≈Fig.5(e) | analytic (was 15.42 under old LRU) |
| het 64→1024 kB | converges (16.07→15.59) | Fig.5: "Het almost constant" ✓ | trend |
| het benefit @64 kB | 20.8 % | text: "up to 79.8% Het vs Hom for ResNet18"* | see note |

\* The paper's "up to 79.8%" is the *Het-vs-Hom* reduction at the data point
where Hom is worst; the 20.8 % printed here is Het-vs-Hom **at 64 kB for
ResNet18 with prefetch+8-bit**, which is the configuration of `main.cpp`. The
abstract's "up to 80%" is the headline across all models/widths. Both are
internally consistent; the harness simply reports one slice.

## Honest labels / scope (not bugs)

1. **Table 3 P1/P3 column swap is a paper typo, not a code error.** §3.2's text
   makes Policy 1 the full-*filter* footprint and Policy 3 the full-*ofmap*
   footprint; Table 3 prints the two columns in the opposite order. The code
   follows the §3.2 formulas, so it labels P1=2318 / P3=788.6. Hand-recomputation
   above confirms the formulas (conv5 → 2318, conv1 → 788.6). The **value set**
   `{2353, 2318, 199.7, 788.6}` is reproduced exactly.

2. **Units are MiB/kiB (1024-based), printed as "MB"/"kB".** `main.cpp`:
   `KB=1024`, `MB=1024²`. Internally consistent and matches the paper's apparent
   convention; the labels are loose only in the SI sense. (Same caveat noted in
   `[[smm-verify-start]]`.)

3. **Off-chip volume counts read *and* write traffic, "transferred once
   from/to".** `estimate_accesses` sums `ifmap + filter + ofmap` (×reload for
   ifmap on P4/P5); the ofmap term is the one-time output write. This matches
   §3.2 ("each element transferred only once from/to off-chip"). It is *not* the
   Stash notion of dirty-writeback/coherence — SMM has no re-write or coherence
   model; an ofmap is a single one-way transfer. (Clarifies the `model-inventory`
   note "SMM has no output-write traffic": it has no *coherence* traffic, but a
   one-time ofmap write *is* counted.)

4. **No explicit energy model.** Off-chip accesses are the energy proxy (§2.2:
   off-chip transfers are 10–100× the energy of a local op). `smm_bench` prints
   footprint/accesses/latency only — consistent with the paper, which reports
   accesses (not joules). (Energy numbers in `model-inventory` come from the
   separate cachemere accelerator bench, not this unit.)

5. **No SCALE-Sim baseline.** The paper's latency baseline is a full SCALE-Sim
   systolic-array simulation; this model implements only the proposed scheme's
   roofline latency estimate (`max`/`sum` of compute vs transfer cycles), which
   is the part that validates the *technique*. The absolute baseline bars of
   Fig. 8 are out of scope.

6. **Only ResNet18 is modeled** (the paper's validation anchor — only network
   with a published per-layer breakdown). The other five Table-2 networks and
   depth-wise convolution (group-aware filter sizing) are not implemented.
   ResNet18 has no DW layers, so this does not affect any reproduced number.
   (Stated in `README.md` §Scope.)

## Correctness checks performed

- All five policy footprint formulas matched term-by-term against §3.2 text.
- Eq. 1 / Eq. 2 ⇔ `mult∈{1,2}` algebra verified (`mult·(I+F+O) ≤ glb`).
- `estimate_accesses` = §3.2 closed form (`once`, P4/P5 ifmap×⌈F#/n⌉); no LRU.
- `choose_block` monotonicity assumption holds: `memory_elems(P4/P5,n)` is
  linear-increasing in `n`, so the early `break` is safe.
- Feasibility: at 64 kB every layer is feasible under P5(n=1) (conv1 ≈14 kB),
  so `heterogeneous` never falls back; homogeneous picks P5 (P2 infeasible at
  64 kB since conv1's full ifmap = 163 kB). Matches printed `Hom 20.28 MB`.
- Table 3 P1/P3 hand-recomputed from the conv5/conv1 hyperparameters → exact.
- Cross-check: `gemm_smm.cpp` (independent self-contained analytic impl) and
  `gemm_policy.cpp` (this unit via `manager.h`) agree on every GEMM shape × GLB.

**Bottom line:** the SMM unit is a faithful analytical reproduction of the
ICPP '24 technique for ResNet18, now with **no LRU/cache** (closed-form §3.2
accesses + §3.3 Algorithm 1) and zero cachemere/abseil/boost dependency. No
correctness bugs found; all deviations from the paper are documented scope
choices or a documented paper typo.
