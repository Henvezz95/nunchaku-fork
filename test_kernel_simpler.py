import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from time import perf_counter

# This script now requires the nunchaku._C library to be compiled and installed.
import nunchaku._C as nunchaku_C

def ceil_div(a, b):
    return (a + b - 1) // b

class SVDQuantLinear(nn.Module):
    """
    Python wrapper for the Nunchaku W4A4+LoRA fused GEMM kernel.
    Includes a classmethod to convert a standard nn.Linear layer.
    """
    def __init__(self, in_features, out_features, lora_rank, bias=True, use_fp4=False, device='cuda:0'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.bias_enabled = bias
        self.device = torch.device(device)
        self.dtype = torch.float16

        print("Initializing Nunchaku backend.")
        self.backend = nunchaku_C.QuantizedGEMM()
        self.backend.init(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias_enabled,
            use_fp4=use_fp4,
            bf16=(self.dtype == torch.bfloat16),
            deviceId=self.device.index,
            rank=self.lora_rank
        )

    def load_weights(self, state_dict):
        processed_dict = {k: v.to(self.device).contiguous() for k, v in state_dict.items()}
        self.backend.loadDict(processed_dict, False)
        print("Weights loaded into the backend.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.dtype).contiguous()
        in_features_pad = math.ceil(self.in_features / 128) * 128
        
        if x.shape[-1] != in_features_pad:
            x_padded = F.pad(x, (0, in_features_pad - self.in_features))
        else:
            x_padded = x

        output_padded = self.backend.forward(x_padded)
        return output_padded[:, :self.out_features]
    

if __name__ == '__main__':
    # Configuration
    in_features = 1024
    out_features = 2048
    lora_rank = 32
    batch_size = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    if device == 'cpu':
        print("ERROR: Nunchaku kernels require a CUDA-enabled GPU. Aborting verification.")
    else:
        print(f"Running verification on device: {device}")

    input_matrix = torch.randint(-7,8,(batch_size, in_features, out_features))
    R = torch.randint(-7,8,(in_features, out_features))
    scales = torch.ones((in_features))
    lora_up = torch.randint(-7,8,(out_features, lora_rank))
    lora_down = torch.randint(-7,8,(in_features, lora_rank))
    qweight, wscales = nunchaku_C.ops.quantize_w4a4_wgt(R.to(dtype))
    state_dict = {
            'qweight': qweight,
            'wscales': wscales,
            'lora_down': lora_down,
            'lora_up': lora_up,
            'smooth': scales,
        }