#pragma once
#include <torch/extension.h>
#include <tuple>


namespace nunchaku::ops {

// Declaration for the weight quantizer function
std::tuple<torch::Tensor, torch::Tensor> quantize_w4a4_wgt(torch::Tensor input);

// Declaration for the activation quantizer function
std::tuple<torch::Tensor, torch::Tensor> quantize_w4a4_act(torch::Tensor input);

void attention_fp16(torch::Tensor q, // packed [Batch, Head, TokensQ, HEAD_DIM]
                    torch::Tensor k, // packed [Batch, Head, TokensKV, HEAD_DIM]
                    torch::Tensor v, // packed [Batch, Head, TokensKV, HEAD_DIM]
                    torch::Tensor o, // linear [Batch, TokensQ, Head * HEAD_DIM]
                    float scale);

torch::Tensor gemv_awq(torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       torch::Tensor _zeros,
                       int64_t m,
                       int64_t n,
                       int64_t k,
                       int64_t group_size);

torch::Tensor
gemm_awq(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros);

// Our new function with a cleaner signature for SVDQuant layers
torch::Tensor gemm_w4a4_lr_fp16(
    const torch::Tensor& act,
    const torch::Tensor& wgt,
    const torch::Tensor& ascales,
    const torch::Tensor& wscales,
    const torch::Tensor& bias,
    const torch::Tensor& lora_down,
    const torch::Tensor& lora_up
);

void test_rmsnorm_rope(torch::Tensor input, torch::Tensor output, torch::Tensor norm_q, torch::Tensor norm_k, torch::Tensor rotary_emb);

void test_pack_qkv(torch::Tensor input, torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v, int numTokens); 

}; // namespace nunchaku::ops
