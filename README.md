# SMM — Scratchpad Memory Management for DL Accelerators

A faithful implementation of the memory-management technique from

> S. Zouzoula, M. A. Maleki, M. W. Azhar, P. Trancoso,
> **"Scratchpad Memory Management for Deep Learning Accelerators"**,
> ICPP '24, pp. 629–639. doi:[10.1145/3673038.3673115](http://dx.doi.org/10.1145/3673038.3673115)
> (PDF: `paper/542648_Fulltext.pdf`).

The on-chip Global Buffer (GLB) is a **software-managed scratchpad**, exactly as
in the paper — *not* a hardware cache. The off-chip access volume is the
**closed-form estimate of Section 3.2** (every data type transferred once, the
partial policies P4/P5 re-loading the ifmap ×⌈F#/n⌉), and Algorithm 1 (Sec 3.3)
only ever selects policies whose footprint fits the GLB. There is **no LRU / no
cache model** — the paper does not use one.

## Paper → code map

| Paper element | Code |
|---|---|
| Layer hyperparameters (Table 1) | `include/smm/layer.h` (`Layer`) |
| On-chip memory policies 1–5 + intra-layer (Sec 3.2) | `include/smm/policy.h` |
| GLB constraint Eq. (1) / Eq. (2) | `evaluate()` in `include/smm/manager.h` (`mult = 1 or 2`) |
| `estimate_memory` / `estimate_accesses` / `estimate_latency` | `memory_elems` / `estimate_accesses` / `evaluate` |
| Algorithm 1 (per-layer policy choice) | `best_for_layer()` |
| Homogeneous / Heterogeneous schemes (Sec 3.3) | `homogeneous()` / `heterogeneous()` |
| GLB partitioning + tile schedule (Fig. 1 / dataflow) | `include/smm/schedule.h` (`build_layout` / `simulate_layer`) |
| ResNet18 model (Table 2) | `include/smm/models.h` (`resnet18()`) |

### Policy definitions (Section 3.2), in elements

| Policy | I_Tile | F_Tile | O_Tile | reloads |
|---|---|---|---|---|
| intra-layer | full ifmap | full filters | full ofmap | ×1 |
| 1 ifmap reuse | `F_H·I_W·C_I` (sliding window) | full filters | `O_W·C_O` | ×1 |
| 2 filter reuse | full ifmap | `F_H·F_W·C_I` (one filter) | `O_H·O_W` | ×1 |
| 3 per-channel | `F_H·I_W` (1 channel) | `F_H·F_W·F#` (1 ch of each filter) | full ofmap | ×1 |
| 4 partial ifmap | sliding window | `F_H·F_W·C_I·n` | `O_W·n` | ifmap ×⌈F#/n⌉ |
| 5 partial per-ch | `F_H·I_W` | `F_H·F_W·n` | `O_H·O_W·n` | ifmap ×⌈F#/n⌉ |

`estimate_accesses` is the closed form of the table above: for the non-partial
policies every element is fetched **once** (`ifmap + filter + ofmap`); the
partial policies P4/P5 add the only reload term the paper defines, `ifmap ×
⌈F#/n⌉`. Algorithm 1 evaluates only policies whose footprint ≤ GLB, so the
"transferred once" assumption holds by construction and no eviction model is
needed. This is why the heterogeneous scheme prefers the larger-footprint
policies whenever they fit.

## Explicit management mechanism (`smm_schedule`)

`policy.h`/`manager.h` answer *what* to do (which policy, what footprint, how many
accesses — the closed form). `include/smm/schedule.h` shows *how* the scratchpad is
actually managed:

- **`build_layout`** physically partitions the GLB into `ifmap` / `filter` /
  `ofmap` regions at byte offsets (Fig. 1; ×2 buf-A/buf-B per region when double
  buffering, Eq. 2).
- **`simulate_layer`** runs the policy's loop nest as a stream of mvin / compute /
  mvout events, accumulating the off-chip volume, peak occupancy, and a
  software-pipelined latency (next tile loads while the current one computes).

`./build/smm_schedule` prints, for each ResNet18 layer, the chosen policy, the GLB
partition map, and the schedule it ran — e.g.:

```
================ conv5_1 : policy2 ================
layer  : 7x7 x512 ifmap, 3x3 x512 x512 filters, stride 1 pad 1

-- GLB partition map (capacity 64.0 kB) --
  [  0.00 ..  24.50] kB  ifmap
  [ 24.50 ..  29.00] kB  filter
  [ 29.00 ..  29.05] kB  ofmap
  used 29.05 / 64.0 kB  (45% utilized)

-- schedule (tiles streamed through the partition) --
  off-chip total : 2353.0 kB   (analytical 2353.0 kB -> MATCH)
  latency: serial 602176 -> pipelined 453152  (25% hidden by double buffering)
```

The simulated off-chip volume equals `estimate_accesses` for **every** layer, so
the schedule is a faithful refinement of the analytical model, not a second guess.

## Build & run

```bash
./build.sh                # g++ -std=c++17, standard library only (no cachemere/boost/abseil)
./build/smm_bench         # Table 3 + access/latency sweeps (analytical)
./build/smm_schedule      # GLB partition map + tile schedule (explicit mechanism)
```

Dependencies: `g++ ≥ 11` and the C++ standard library — nothing else. (Earlier
revisions modeled the GLB with a cachemere LRU cache and pulled in boost/abseil;
all of that was removed because the paper's scratchpad is SW-managed and its
off-chip volume is closed-form, so no eviction model is needed.)

## Validation

A line-by-line audit of this model against the paper is in
[`VERIFICATION.md`](VERIFICATION.md). In short, `smm_bench` reproduces **Table 3**
(max per-policy footprint, kB) for ResNet18:

| | intra | full-ifmap | full-ofmap | full-filter |
|---|---|---|---|---|
| this code | 2353.0 | 199.6 (P2) | 788.6 (P3) | 2318.0 (P1) |
| paper Table 3 | 2353.0 | 199.7 (P2) | 788.6 (**P1**) | 2318.0 (**P3**) |

The value set matches exactly. The textual policy definitions make Policy 1 the
full-filter footprint (~2318) and Policy 3 the full-ofmap footprint (~788.6);
Table 3 prints the P1/P3 columns in the opposite order, which is a typo in the
table — the formulas in §3.2 are unambiguous.

It also reproduces the qualitative trends of Figs. 5/7: the heterogeneous scheme
beats the homogeneous one most at small buffers (≈21% fewer off-chip accesses at
64 kB) and converges as the buffer grows (0% by 256 kB), while the heterogeneous
off-chip access volume stays almost constant across buffer sizes (§5.1).

## Scope

ResNet18 is implemented in full because the paper publishes its per-layer
breakdown, making it the validation anchor. The other Table-2 networks can be
added as `Model`s of `Layer` descriptions; depth-wise convolution (DW) would need
group-aware filter sizing in `policy.h` and is not modeled here.
