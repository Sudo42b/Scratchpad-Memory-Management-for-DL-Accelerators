#ifndef SMM_MODELS_H
#define SMM_MODELS_H

#include "smm/layer.h"

// Reference model definitions (Table 2).  ResNet18 is given in full because the
// paper reports its per-layer memory breakdown (Figs. 3 & 6) and its Table 3
// numbers, which makes it the validation anchor for this implementation.

namespace smm {

namespace detail {

// Convenience constructor for a convolutional layer.
inline Layer conv(const std::string& name, i64 ih, i64 ci, i64 fh, i64 fn, i64 s, i64 p, LayerType t = LayerType::CV)
{
    Layer L;
    L.name = name;
    L.type = t;
    L.IH = L.IW = ih;
    L.FH = L.FW = fh;
    L.CI = ci;
    L.Fn = fn;
    L.S  = s;
    L.P  = p;
    return L;
}

}  // namespace detail

// ResNet18 (He et al., 2016).  21 weight layers: conv1, 16 block convs,
// 3 projection (1x1 downsample) layers, and the final fully-connected layer.
inline Model resnet18()
{
    using detail::conv;
    Model m;
    m.name = "ResNet18";
    auto& v = m.layers;

    // stem: 224x224x3 -> 112x112x64, then 3x3/2 maxpool -> 56x56x64
    v.push_back(conv("conv1", 224, 3, 7, 64, 2, 3));

    // conv2_x : 56x56x64, four 3x3/64
    for (int i = 0; i < 4; ++i) v.push_back(conv("conv2_" + std::to_string(i), 56, 64, 3, 64, 1, 1));

    // conv3_x : down to 28x28x128
    v.push_back(conv("conv3_0", 56, 64, 3, 128, 2, 1));                        // stride-2 entry
    v.push_back(conv("conv3_ds", 56, 64, 1, 128, 2, 0, LayerType::PL));        // projection shortcut
    for (int i = 1; i < 4; ++i) v.push_back(conv("conv3_" + std::to_string(i), 28, 128, 3, 128, 1, 1));

    // conv4_x : down to 14x14x256
    v.push_back(conv("conv4_0", 28, 128, 3, 256, 2, 1));
    v.push_back(conv("conv4_ds", 28, 128, 1, 256, 2, 0, LayerType::PL));
    for (int i = 1; i < 4; ++i) v.push_back(conv("conv4_" + std::to_string(i), 14, 256, 3, 256, 1, 1));

    // conv5_x : down to 7x7x512
    v.push_back(conv("conv5_0", 14, 256, 3, 512, 2, 1));
    v.push_back(conv("conv5_ds", 14, 256, 1, 512, 2, 0, LayerType::PL));
    for (int i = 1; i < 4; ++i) v.push_back(conv("conv5_" + std::to_string(i), 7, 512, 3, 512, 1, 1));

    // global average pool -> fully connected 512 -> 1000 (as a 1x1 conv)
    v.push_back(conv("fc", 1, 512, 1, 1000, 1, 0, LayerType::FC));

    return m;
}

}  // namespace smm

#endif  // SMM_MODELS_H
