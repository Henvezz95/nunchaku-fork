#include "gemm.h"
#include "gemm88.h"
#include "flux.h"
#include "sana.h"
#include "ops.h"
#include "utils.h"
#include <torch/extension.h>
#include "interop/torch.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for std::map support
#include "Linear.h"
#include "FusedMLP.h"


std::map<std::string, Tensor> torch_dict_to_nunchaku_map(const std::map<std::string, torch::Tensor>& dict) {
    std::map<std::string, Tensor> nunchaku_map;
    for (const auto& pair : dict) {
        nunchaku_map[pair.first] = from_torch(pair.second.contiguous());
    }
    return nunchaku_map;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<QuantizedFluxModel>(m, "QuantizedFluxModel")
        .def(py::init<>())
        .def("init",
             &QuantizedFluxModel::init,
             py::arg("use_fp4"),
             py::arg("offload"),
             py::arg("bf16"),
             py::arg("deviceId"))
        .def("set_residual_callback",
             [](QuantizedFluxModel &self, pybind11::object call_back) {
                 if (call_back.is_none()) {
                     self.set_residual_callback(pybind11::function());
                 } else {
                     self.set_residual_callback(call_back);
                 }
             })
        .def("reset", &QuantizedFluxModel::reset)
        .def("load", &QuantizedFluxModel::load, py::arg("path"), py::arg("partial") = false)
        .def("loadDict", &QuantizedFluxModel::loadDict, py::arg("dict"), py::arg("partial") = false)
        .def("forward",
             &QuantizedFluxModel::forward,
             py::arg("hidden_states"),
             py::arg("encoder_hidden_states"),
             py::arg("temb"),
             py::arg("rotary_emb_img"),
             py::arg("rotary_emb_context"),
             py::arg("rotary_emb_single"),
             py::arg("controlnet_block_samples")        = py::none(),
             py::arg("controlnet_single_block_samples") = py::none(),
             py::arg("skip_first_layer")                = false)
        .def("forward_layer",
             &QuantizedFluxModel::forward_layer,
             py::arg("idx"),
             py::arg("hidden_states"),
             py::arg("encoder_hidden_states"),
             py::arg("temb"),
             py::arg("rotary_emb_img"),
             py::arg("rotary_emb_context"),
             py::arg("controlnet_block_samples")        = py::none(),
             py::arg("controlnet_single_block_samples") = py::none())
        .def("forward_layer_ip_adapter",
             &QuantizedFluxModel::forward_layer_ip_adapter,
             py::arg("idx"),
             py::arg("hidden_states"),
             py::arg("encoder_hidden_states"),
             py::arg("temb"),
             py::arg("rotary_emb_img"),
             py::arg("rotary_emb_context"),
             py::arg("controlnet_block_samples")        = py::none(),
             py::arg("controlnet_single_block_samples") = py::none())
        .def("forward_single_layer", &QuantizedFluxModel::forward_single_layer)
        .def("norm_one_forward", &QuantizedFluxModel::norm_one_forward)
        .def("startDebug", &QuantizedFluxModel::startDebug)
        .def("stopDebug", &QuantizedFluxModel::stopDebug)
        .def("getDebugResults", &QuantizedFluxModel::getDebugResults)
        .def("setLoraScale", &QuantizedFluxModel::setLoraScale)
        .def("setAttentionImpl", &QuantizedFluxModel::setAttentionImpl)
        .def("isBF16", &QuantizedFluxModel::isBF16);
    py::class_<QuantizedSanaModel>(m, "QuantizedSanaModel")
        .def(py::init<>())
        .def("init",
             &QuantizedSanaModel::init,
             py::arg("config"),
             py::arg("pag_layers"),
             py::arg("use_fp4"),
             py::arg("bf16"),
             py::arg("deviceId"))
        .def("reset", &QuantizedSanaModel::reset)
        .def("load", &QuantizedSanaModel::load, py::arg("path"), py::arg("partial") = false)
        .def("loadDict", &QuantizedSanaModel::loadDict, py::arg("dict"), py::arg("partial") = false)
        .def("forward", &QuantizedSanaModel::forward)
        .def("forward_layer", &QuantizedSanaModel::forward_layer)
        .def("startDebug", &QuantizedSanaModel::startDebug)
        .def("stopDebug", &QuantizedSanaModel::stopDebug)
        .def("getDebugResults", &QuantizedSanaModel::getDebugResults);
    py::class_<QuantizedGEMM>(m, "QuantizedGEMM")
        .def(py::init<>())
        .def("init", &QuantizedGEMM::init, 
            py::arg("in_features"), 
            py::arg("out_features"), 
            py::arg("bias"), 
            py::arg("use_fp4"), 
            py::arg("bf16"), 
            py::arg("deviceId"), 
            py::arg("rank") = 0)
        .def("reset", &QuantizedGEMM::reset)
        .def("load", &QuantizedGEMM::load)
        .def("loadDict", &QuantizedGEMM::loadDict, py::arg("dict"), py::arg("partial") = false) // <-- ADD THIS LINE
        .def("forward", &QuantizedGEMM::forward)
        .def("quantize", &QuantizedGEMM::quantize)
        .def("startDebug", &QuantizedGEMM::startDebug)
        .def("stopDebug", &QuantizedGEMM::stopDebug)
        .def("getDebugResults", &QuantizedGEMM::getDebugResults);
    py::class_<QuantizedFusedMLP>(m, "QuantizedFusedMLP")
        .def(py::init<>())
        .def("init",
            &QuantizedFusedMLP::init,
            py::arg("in_features"),
            py::arg("hidden_features"),
            py::arg("bias"),
            py::arg("use_fp4"),
            py::arg("bf16"),
            py::arg("deviceId"))
        .def("loadDict",
            &QuantizedFusedMLP::loadDict,
            py::arg("state_dict"))
        .def("forward",
            &QuantizedFusedMLP::forward,
            py::arg("x"));
    
    //py::class_<Tensor>(m, "Tensor");
    // Give the Tensor class binding a variable name so we can attach the enum to it
    py::class_<Tensor> tensor_class(m, "Tensor");

    // Define and export the ScalarType enum as a nested type within the Tensor class
    py::enum_<Tensor::ScalarType>(tensor_class, "ScalarType")
        .value("FP32",     Tensor::ScalarType::FP32)
        .value("FP16",     Tensor::ScalarType::FP16)
        .value("BF16",     Tensor::ScalarType::BF16)
        .value("INT32",    Tensor::ScalarType::INT32)
        .value("INT8",     Tensor::ScalarType::INT8)
        .value("FP8_E4M3", Tensor::ScalarType::FP8_E4M3)
        .export_values(); // This makes the enum members accessible
    
    /*py::class_<GEMM_W4A4>(m, "QuantizedGEMM_W4A4")
        .def(py::init<int, int, bool, bool, Tensor::ScalarType, Device>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("bias"),
             py::arg("use_fp4"),
             py::arg("dtype"),
             py::arg("device"))
        .def("forward", [](GEMM_W4A4 &self, torch::Tensor x) {
            Tensor nunchaku_x = from_torch(x.contiguous());
            Tensor nunchaku_out = self.forward(nunchaku_x);
            return to_torch(nunchaku_out);
        }, py::arg("x"))
        .def("load_from_dict", [](GEMM_W4A4 &self, const std::map<std::string, torch::Tensor>& dict, bool partial) {
            auto nunchaku_map = torch_dict_to_nunchaku_map(dict);
            self.load_from_dict(nunchaku_map, partial);
        }, py::arg("dict"), py::arg("partial") = false);*/



    m.def_submodule("ops")
        .def("attention_fp16", nunchaku::ops::attention_fp16)
        .def("gemm_awq", nunchaku::ops::gemm_awq)
        .def("gemv_awq", nunchaku::ops::gemv_awq)

        .def("test_rmsnorm_rope", nunchaku::ops::test_rmsnorm_rope)
        .def("test_pack_qkv", nunchaku::ops::test_pack_qkv)
        .def("quantize_w4a4_wgt", [](torch::Tensor input) {
            auto result_tuple = nunchaku::ops::quantize_w4a4_wgt(input);
            return std::make_tuple(std::get<0>(result_tuple), std::get<1>(result_tuple));
        }, py::arg("input"))
        .def("quantize_w4a4_act", [](torch::Tensor input) {
            auto result_tuple = nunchaku::ops::quantize_w4a4_act(input);
            return std::make_tuple(std::get<0>(result_tuple), std::get<1>(result_tuple));
        }, py::arg("input"));

    m.def_submodule("utils")
        .def("set_log_level", [](const std::string &level) { spdlog::set_level(spdlog::level::from_str(level)); })
        .def("set_cuda_stack_limit", nunchaku::utils::set_cuda_stack_limit)
        .def("disable_memory_auto_release", nunchaku::utils::disable_memory_auto_release)
        .def("trim_memory", nunchaku::utils::trim_memory)
        .def("set_faster_i2f_mode", nunchaku::utils::set_faster_i2f_mode);
}