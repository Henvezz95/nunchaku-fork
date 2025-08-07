#pragma once

#include "Linear.h"
#include "Tensor.h"
#include "interop/torch.h"
#include <torch/extension.h>

// Final stateless wrapper function
inline torch::Tensor gemm_w4a4_svd(
    const torch::Tensor& act,
    const torch::Tensor& qweight,
    const torch::Tensor& wscales,
    const torch::Tensor& bias,
    const torch::Tensor& lora_down,
    const torch::Tensor& lora_up,
    const torch::Tensor& smooth
) {
    int in_features = act.size(1);
    int out_features = qweight.size(0);
    bool use_bias = bias.defined();
    
    Tensor::ScalarType dtype;
    switch (act.scalar_type()) {
        case at::ScalarType::Half:
            dtype = Tensor::ScalarType::FP16;
            break;
        case at::ScalarType::BFloat16:
            dtype = Tensor::ScalarType::BF16;
            break;
        default:
            throw std::runtime_error("Unsupported input dtype for gemm_w4a4_svd");
    }
    Device device(Device::CUDA, act.device().index());

    GEMM_W4A4 gemm(in_features, out_features, use_bias, false, dtype, device);

    std::map<std::string, Tensor> state_dict;
    state_dict["qweight"] = from_torch(qweight);
    state_dict["wscales"] = from_torch(wscales);
    if (use_bias) state_dict["bias"] = from_torch(bias);
    state_dict["lora_down"] = from_torch(lora_down);
    state_dict["lora_up"] = from_torch(lora_up);
    state_dict["smooth"] = from_torch(smooth);
    
    // Corrected: Call the public loading method we added to the GEMM_W4A4 class
    gemm.load_from_dict(state_dict, true);

    auto q_act = gemm.quantize(from_torch(act), false);
    auto result_variant = gemm.forward_quant(q_act, GEMM_W4A4::FuseOptions::EMPTY, nullptr);

    return to_torch(std::get<Tensor>(result_variant));
}