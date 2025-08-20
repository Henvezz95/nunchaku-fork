#include "FusedMLP.h"

// Now, in the .cpp file, we include the full definition for GEMM_W4A4
// so we can create and use it.
#include "Linear.h"
#include <variant>

// Constructor and Destructor need to be here, after GEMM_W4A4 is fully defined.
QuantizedFusedMLP::QuantizedFusedMLP() = default;
QuantizedFusedMLP::~QuantizedFusedMLP() = default;

// --- Initialization Method ---
void QuantizedFusedMLP::init(int64_t in_features, int64_t hidden_features, bool bias, bool use_fp4, bool bf16, int8_t deviceId) {
    auto dtype  = bf16 ? Tensor::BF16 : Tensor::FP16;
    auto device = Device::cuda((int)deviceId);

    fc1 = std::make_unique<GEMM_W4A4>((int)in_features, (int)hidden_features, bias, use_fp4, dtype, device);
    fc2 = std::make_unique<GEMM_W4A4>((int)hidden_features, (int)in_features, bias, use_fp4, dtype, device);
}

// --- Weight Loading Method ---
 void QuantizedFusedMLP::loadDict(const std::map<std::string, torch::Tensor>& dict) {
     std::map<std::string, Tensor> fc1_dict;
     std::map<std::string, Tensor> fc2_dict;
     for (const auto &kv : dict) {
         const auto &key = kv.first;
         const auto &val = kv.second.contiguous();
         Tensor t = from_torch(val);
         if (key.rfind("fc1.", 0) == 0) {
             fc1_dict.emplace(key.substr(4), t);
         } else if (key.rfind("fc2.", 0) == 0) {
             fc2_dict.emplace(key.substr(4), t);
         }
     }
     fc1->load_from_dict(fc1_dict, /*partial=*/false);
     fc2->load_from_dict(fc2_dict, /*partial=*/false);
 }

// --- Forward Pass Method ---
torch::Tensor QuantizedFusedMLP::forward(torch::Tensor x) {
    TORCH_CHECK(fc1 && fc2, "QuantizedFusedMLP has not been initialized. Call .init() before forward().");

    Tensor nunchaku_x = from_torch(x.contiguous());

    auto fused_variant = fc1->forward(nunchaku_x,
                                      GEMM_W4A4::FuseOptions::GELU_QUANT,
                                      fc2.get());

    auto quantized_activation = std::get<GEMM_W4A4::QuantizedActivation>(fused_variant);

    Tensor nunchaku_out = fc2->forward_quant(quantized_activation);

    return to_torch(nunchaku_out);
}