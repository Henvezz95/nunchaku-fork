#pragma once

// Standard library includes needed for the class definition
#include <memory>
#include <map>

// PyTorch and Nunchaku base class includes
#include "interop/torch.h"
#include "Module.h"

// Forward-declare the classes we hold pointers to.
// This avoids including the entire Linear.h and breaks the circular dependency.
class GEMM_W4A4;

class QuantizedFusedMLP : public Module {
public:
    // Constructor & Destructor
    QuantizedFusedMLP();
    ~QuantizedFusedMLP();

    // Public methods to be called from Python
    void init(int64_t in_features, int64_t hidden_features, bool bias, bool use_fp4, bool bf16, int8_t deviceId);
    void loadDict(const std::map<std::string, torch::Tensor>& dict);
    torch::Tensor forward(torch::Tensor x);

private:
    // std::unique_ptr only requires a forward declaration in the header,
    // which makes this possible.
    std::unique_ptr<GEMM_W4A4> fc1;
    std::unique_ptr<GEMM_W4A4> fc2;
};