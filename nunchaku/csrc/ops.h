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

void gemm_w4a4(
    std::optional<torch::Tensor> act,
    std::optional<torch::Tensor> wgt,
    std::optional<torch::Tensor> out,
    std::optional<torch::Tensor> qout,
    std::optional<torch::Tensor> ascales,
    std::optional<torch::Tensor> wscales,
    std::optional<torch::Tensor> oscales,
    std::optional<torch::Tensor> poolout,
    std::optional<torch::Tensor> lora_act_in,
    std::optional<torch::Tensor> lora_up,
    std::optional<torch::Tensor> lora_down,
    std::optional<torch::Tensor> lora_act_out,
    std::optional<torch::Tensor> norm_q,
    std::optional<torch::Tensor> norm_k,
    std::optional<torch::Tensor> rotary_emb,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> smooth_factor,
    std::optional<torch::Tensor> out_vk,
    std::optional<torch::Tensor> out_linearattn,
    bool act_unsigned,
    std::vector<float> lora_scales,
    bool fuse_silu,
    bool fp4,
    float alpha,
    std::optional<torch::Tensor> wcscales,
    std::optional<torch::Tensor> out_q,
    std::optional<torch::Tensor> out_k,
    std::optional<torch::Tensor> out_v,
    int attn_tokens
);

void gemm_w4a4_dummy(int a, int b);

}; // namespace nunchaku::ops
