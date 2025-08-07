import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from time import perf_counter

# This script now requires the nunchaku._C library to be compiled and installed.
import nunchaku._C as nunchaku_C

# --- The Python Wrapper Module ---

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

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, lora_rank: int, alpha: float = 0.5):
        """
        Creates and initializes an SVDQuantLinear layer from a standard torch.nn.Linear layer.
        This method performs the full SVDQuant offline process.
        """
        if not isinstance(linear_layer, nn.Linear):
            raise TypeError("Input must be an instance of torch.nn.Linear")

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        use_bias = linear_layer.bias is not None
        device = linear_layer.weight.device
        dtype = linear_layer.weight.dtype

        svd_linear = cls(in_features, out_features, lora_rank, use_bias, device=str(device))

        print(f"\n--- Converting nn.Linear({in_features}, {out_features}) to SVDQuantLinear ---")
        
        # 1. Prepare weights
        original_weight = linear_layer.weight.data.clone().to(torch.float32)
        bias = linear_layer.bias.data.clone().to(dtype) if use_bias else None

        # 2. Perform Smoothing
        print("Using random data for activation smoothing calibration...")
        calibration_data = torch.randn(256, in_features, dtype=torch.float32, device=device)
        act_scales = torch.max(torch.abs(calibration_data), dim=0)[0].clamp(min=1e-5)
        
        weight_scales = torch.max(torch.abs(original_weight), dim=0)[0].clamp(min=1e-5)
        
        s = torch.pow(act_scales, alpha) / torch.pow(weight_scales, 1 - alpha)
        
        smoothed_weight = original_weight * s.unsqueeze(0)
        smooth_factor = (1.0 / s).to(dtype)

        # 3. Perform SVD on the smoothed weight
        print("Performing SVD...")
        U, S, Vh = torch.linalg.svd(smoothed_weight, full_matrices=False)

        # 4. Decompose into low-rank matrices with correct math
        sqrt_S_diag = torch.diag(torch.sqrt(S[:lora_rank]))
        
        lora_up = (U[:, :lora_rank] @ sqrt_S_diag).to(dtype)
        lora_down_T = (sqrt_S_diag @ Vh[:lora_rank, :])
        
        # *** FIX: Cast lora_down to the correct dtype before reconstruction ***
        lora_down = lora_down_T.T.to(dtype) # This will be [in_features, rank]

        # 5. Calculate the residual weight
        low_rank_approx_smoothed = (lora_up @ lora_down.T).to(torch.float32)
        residual_smoothed = smoothed_weight - low_rank_approx_smoothed
        residual_weight = residual_smoothed / s.unsqueeze(0)
        
        # 6. Pad all components
        pad_to = 128
        in_features_pad = ceil_div(in_features, pad_to) * pad_to
        out_features_pad = ceil_div(out_features, pad_to) * pad_to

        residual_padded = F.pad(residual_weight, (0, in_features_pad - in_features, 0, out_features_pad - out_features))
        lora_down_padded = F.pad(lora_down, (0, in_features_pad - in_features))
        lora_up_padded = F.pad(lora_up, (0, 0, 0, out_features_pad - out_features))
        smooth_padded = F.pad(smooth_factor, (0, in_features_pad - in_features))
        bias_padded = F.pad(bias, (0, out_features_pad - out_features)) if use_bias else None

        # 7. Quantize the residual weight using the C++ op
        print("Quantizing residual weight...")
        qweight, wscales = nunchaku_C.ops.quantize_w4a4_wgt(residual_padded.to(dtype))

        # 8. Load all prepared tensors into the C++ module
        state_dict = {
            'qweight': qweight,
            'wscales': wscales,
            'lora_down': lora_down_padded.contiguous(),
            'lora_up': lora_up_padded.contiguous(),
            'smooth': smooth_padded,
        }
        if use_bias:
            state_dict['bias'] = bias_padded

        svd_linear.load_weights(state_dict)
        print("Conversion complete.")
        return svd_linear

# --- Verification Script ---

def ceil_div(a, b):
    return (a + b - 1) // b

if __name__ == '__main__':
    # Configuration
    in_features = 1024
    out_features = 2048
    lora_rank = 32
    batch_size = 1024
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    if device == 'cpu':
        print("ERROR: Nunchaku kernels require a CUDA-enabled GPU. Aborting verification.")
    else:
        print(f"Running verification on device: {device}")

        # 1. Create a standard, pre-trained nn.Linear layer
        print("\n--- Creating Original nn.Linear Layer ---")
        linear_layer = nn.Linear(in_features, out_features, bias=True).to(device, dtype)

        # 2. Convert it to our SVDQuantLinear layer
        # The from_linear method handles all the complex conversion steps.
        try:
            svdquant_layer = SVDQuantLinear.from_linear(linear_layer, lora_rank)

            # 3. Create a test input tensor
            x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

            # 4. Run both layers and compare their outputs
            print("\n--- Comparing Layer Outputs ---")
            with torch.no_grad():
                start = perf_counter()
                output_original = linear_layer(x)
                print('Comparison Layer Time:', perf_counter()-start)
                start = perf_counter()
                output_svdquant = svdquant_layer(x)
                print('SVDQuantLayer Time:', perf_counter()-start)

            # 5. Calculate the quantization error
            # This is the most important metric. We expect it to be small, but not zero.
            mse = torch.mean((output_original - output_svdquant).float() ** 2)
            print(f"\nMean Squared Error (MSE) between original and SVDQuant layer: {mse.item():.6f}")

            # Calculate Signal-to-Noise Ratio (SNR) as another quality metric
            snr = 10 * torch.log10(torch.mean(output_original.float()**2) / mse).item()
            print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

            if snr > 30:
                print("\n✅ Verification successful! The quantization error is low, as expected.")
            else:
                print("\n⚠️ Verification warning: The quantization error is higher than expected. This could be due to the random calibration data.")


        except ImportError:
            print("\nERROR: Could not import 'nunchaku._C'.")
            print("Please ensure the C++ library is compiled and the resulting file")
            print("is accessible in your Python environment (e.g., via 'pip install .').")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
