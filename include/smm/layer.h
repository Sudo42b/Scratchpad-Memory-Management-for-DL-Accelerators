#ifndef SMM_LAYER_H
#define SMM_LAYER_H

#include <cstdint>
#include <string>
#include <vector>

// Implementation of "Scratchpad Memory Management for Deep Learning Accelerators"
// (Zouzoula et al., ICPP '24, doi:10.1145/3673038.3673115).
//
// This header models a single model layer with the hyperparameters of Table 1.

namespace smm {

using i64 = std::int64_t;

enum class LayerType { CV, DW, PW, FC, PL };  // Table 2 layer types

// Hyperparameters of a model layer (Table 1 of the paper).
//
//   I_H / I_W : ifmap height / width
//   F_H / F_W : filter height / width
//   C_I       : number of ifmap / filter channels
//   F_#       : number of 3D filters (== C_O, the number of ofmap channels)
//   S         : stride
//   P         : padding
struct Layer {
    std::string name;
    LayerType   type = LayerType::CV;

    i64 IH = 0, IW = 0;  // ifmap height / width
    i64 FH = 0, FW = 0;  // filter height / width
    i64 CI = 0;          // input (ifmap / filter) channels
    i64 Fn = 0;          // number of 3D filters  (F_#)
    i64 S  = 1;          // stride
    i64 P  = 0;          // padding

    // --- derived quantities (Section 2.2 / Table 1) ---
    i64 OH() const { return (IH + 2 * P - FH) / S + 1; }   // ofmap height
    i64 OW() const { return (IW + 2 * P - FW) / S + 1; }   // ofmap width
    i64 CO() const { return Fn; }                          // ofmap channels

    // element counts of each data type
    i64 ifmap_elems()  const { return IH * IW * CI; }
    i64 filter_elems() const { return FH * FW * CI * Fn; }
    i64 ofmap_elems()  const { return OH() * OW() * CO(); }
};

struct Model {
    std::string        name;
    std::vector<Layer> layers;
};

}  // namespace smm

#endif  // SMM_LAYER_H
