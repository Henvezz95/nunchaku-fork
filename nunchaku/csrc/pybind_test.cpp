#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Now include project's headers
#include "gemm.h"
#include "gemm88.h"
#include "flux.h"
#include "sana.h"
#include "ops.h"
#include "utils.h"
#include "interop/torch.h"
#include "Linear.h"

namespace py = pybind11;
using namespace nunchaku;

// Helper function to convert a Python dictionary of tensors to a C++ map
std::map<std.string, Tensor> torch_dict_to_nunchaku_map(const std::map<std::string, torch::Tensor>& dict) {
    std::map<std::string, Tensor> nunchaku_map;
    for (const auto& pair : dict) {
        nunchaku_map[pair.first] = from_torch(pair.second.contiguous());
    }
    return nunchaku_map;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose existing model classes from the library
    py::class_<QuantizedFluxModel>(m, "QuantizedFluxModel")
        .def(py::init<>())
        .def("init", &QuantizedFluxModel::init, py::arg("use_fp4"), py::arg("offload"), py::arg("bf16"), py::arg("deviceId"))
        .def("forward", &QuantizedFluxModel::forward, py::arg("hidden_states"), py::arg("encoder_hidden_states"), py::arg("temb"), py::arg("rotary_emb_img"), py::arg("rotary_emb_context"), py::arg("rotary_emb_single"), py::arg("controlnet_block_samples") = py::none(), py::arg("controlnet_single_block_samples") = py::none(), py::arg("skip_first_layer") = false)
        .def("loadDict", &QuantizedFluxModel::loadDict, py::arg("dict"), py::arg("partial") = false);

    py::class_<QuantizedSanaModel>(m, "QuantizedSanaModel");
    py::class_<QuantizedGEMM>(m, "QuantizedGEMM");
    py::class_<QuantizedGEMM88>(m, "QuantizedGEMM88");
    py::class_<Tensor>(m, "Tensor");

    // Expose the Device and ScalarType enums so they can be used from Python
    py::class_<Device>(m, "Device")
        .def(py::init<Device::Type, int>())
        .def_readonly("type", &Device::type)
        .def_readonly("idx", &Device::idx);

    py::enum_<Device::Type>(m, "DeviceType")
        .value("CUDA", Device::Type::CUDA)
        .value("CPU", Device::Type::CPU);

    py::enum_<Tensor::ScalarType>(m, "ScalarType")
        .value("FP32", Tensor::ScalarType::FP32)
        .value("FP16", Tensor::ScalarType::FP16)
        .value("BF16", Tensor::ScalarType::BF16)
        .value("INT32", Tensor::ScalarType::INT32)
        .value("INT8", Tensor::ScalarType::INT8)
        .value("FP8_E4M3", Tensor::ScalarType::FP8_E4M3);
    
    // Main class binding for our SVDQuant Linear Layer
    py::class_<GEMM_W4A4>(m, "QuantizedGEMM_W4A4")
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
        }, py::arg("dict"), py::arg("partial") = false);


    // Define the "ops" submodule ONCE and add all functions to it.
    auto ops_module = m.def_submodule("ops");

    // Add our new, clean lambda wrappers for tuple returns
    ops_module.def("quantize_w4a4_wgt", [](torch::Tensor input) {
        auto result_tuple = nunchaku::ops::quantize_w4a4_wgt(input);
        return std::make_tuple(
            std::get<0>(result_tuple),
            std::get<1>(result_tuple)
        );
    }, py::arg("input"));

    ops_module.def("quantize_w4a4_act", [](torch::Tensor input) {
        auto result_tuple = nunchaku::ops::quantize_w4a4_act(input);
        return std::make_tuple(
            std::get<0>(result_tuple),
            std::get<1>(result_tuple)
        );
    }, py::arg("input"));

    // Add the original functions from the library to the SAME submodule
    ops_module.def("attention_fp16", &nunchaku::ops::attention_fp16);
    ops_module.def("gemm_awq", &nunchaku::ops::gemm_awq);
    ops_module.def("gemv_awq", &nunchaku::ops::gemv_awq);
    ops_module.def("test_rmsnorm_rope", &nunchaku::ops::test_rmsnorm_rope);
    ops_module.def("test_pack_qkv", &nunchaku::ops::test_pack_qkv);


    // Define the "utils" submodule
    m.def_submodule("utils")
        .def("set_log_level", [](const std::string &level) { spdlog::set_level(spdlog::level::from_str(level)); })
        .def("set_cuda_stack_limit", &nunchaku::utils::set_cuda_stack_limit)
        .def("disable_memory_auto_release", &nunchaku::utils::disable_memory_auto_release)
        .def("trim_memory", &nunchaku::utils::trim_memory)
        .def("set_faster_i2f_mode", &nunchaku::utils::set_faster_i2f_mode);
}