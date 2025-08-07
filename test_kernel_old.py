import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# This script assumes you have successfully compiled the C++ extension and it's
# available in your Python environment.
try:
    import nunchaku._C as nunchaku_C
except ImportError as e:
    print("Fatal Error: Could not import the compiled C++ extension 'nunchaku._C'.")
    print("Please ensure you have run the setup.py build command successfully.")
    print(f"Original error: {e}")
    exit()

nunchaku_C.utils.set_log_level("trace")

class SVDLinear(nn.Module):
    """
    A drop-in replacement for torch.nn.Linear that uses SVDQuant (W4A4 + FP16 Low-Rank)
    for accelerated inference.

    This module encapsulates the logic for weight preparation (SVD, smoothing, quantization)
    and uses the high-performance C++/CUDA backend for the forward pass.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, use_bias: bool = True, device=None, dtype=torch.float16):
        """
        Initializes the SVDLinear layer.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.use_bias = use_bias
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        if self.device.type != 'cuda':
            raise RuntimeError("Nunchaku SVDLinear layer requires a CUDA device.")

        # 1. Create an instance of the high-level C++ wrapper
        self.cpp_module = nunchaku_C.QuantizedGEMM()

        # 2. Call its init method to configure the underlying engine
        is_bf16 = (self.dtype == torch.bfloat16)
        self.cpp_module.init(
            self.in_features,
            self.out_features, 
            self.use_bias,
            False,  # use_fp4
            is_bf16,
            self.device.index,
            self.rank
        )

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, rank: int, input_tensor_for_test: torch.Tensor, alpha: float = 0.5):
        """
        Creates and initializes an SVDLinear layer from a standard torch.nn.Linear layer.

        This method performs the SVD decomposition, smoothing, and weight quantization.
        """
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        use_bias = linear_layer.bias is not None
        device = linear_layer.weight.device
        dtype = linear_layer.weight.dtype

        svd_linear = cls(in_features, out_features, rank, use_bias, device, dtype)

        # 1. Prepare weights from the original layer
        original_weight = linear_layer.weight.data.clone().to(torch.float32)
        bias = linear_layer.bias.data.clone().to(dtype) if use_bias else None

        # 2. Perform Smoothing (as described in the SVDQuant paper)
        # For a robust implementation, you'd use a small, representative calibration dataset.
        # For this test, we'll simulate it with random data.
        print("Note: Using random data for activation smoothing calibration. For best results, use a real calibration dataset.")
        calibration_data = torch.randn(256, in_features, dtype=torch.float32, device=device)
        
        act_scales = torch.max(torch.abs(calibration_data), dim=0)[0]
        weight_scales = torch.max(torch.abs(original_weight), dim=0)[0]
        
        # Prevent division by zero
        act_scales.clamp_(min=1e-5)
        weight_scales.clamp_(min=1e-5)
        
        # Calculate smoothing factor `s`
        s = torch.pow(act_scales, alpha) / torch.pow(weight_scales, 1 - alpha)
        
        # Apply smoothing to the weight: W' = W * diag(s)
        smoothed_weight = original_weight * s.unsqueeze(0)
        
        # The inverted smoothing factor will be applied to the activations at runtime
        smooth_factor = (1.0 / s).to(dtype)

        # 3. Perform SVD on the smoothed weight
        U, S, Vh = torch.linalg.svd(smoothed_weight, full_matrices=False)

        # 4. Decompose into low-rank matrices (lora_down and lora_up in Nunchaku)
        lora_up = (U[:, :rank] * torch.sqrt(S[:rank]).unsqueeze(0)).to(dtype).contiguous()
        lora_down = (Vh[:rank, :].T * torch.sqrt(S[:rank]).unsqueeze(0)).to(dtype).contiguous()

        # 5. Calculate the residual weight matrix (in original, unsmoothed space)
        # 5a. Reconstruct the low-rank approximation of the *smoothed* weight
        low_rank_approx_smoothed = lora_up @ lora_down.T

        # 5b. Calculate the residual in the *smoothed* space
        residual_smoothed = smoothed_weight - low_rank_approx_smoothed

        # 5c. Transform the residual back to the *original* weight space before quantization.
        #    (We smoothed by multiplying by s, so we un-smooth by dividing by s).
        residual_weight = residual_smoothed / s.unsqueeze(0)
        
        # 6. Quantize the residual weight using the C++ backend op
        qweight, wscales = nunchaku_C.ops.quantize_w4a4_wgt(residual_weight.to(dtype))

        # 7. Load all the prepared tensors into the stateful C++ module
        state_dict = {
            "qweight": qweight,
            "wscales": wscales,
            "lora_down": lora_down,
            "lora_up": lora_up,
            "smooth": smooth_factor,
        }
        if use_bias and bias is not None:
            state_dict["bias"] = bias

        svd_linear.cpp_module.loadDict(state_dict, False)

        print(f"Successfully converted nn.Linear({in_features}, {out_features}) to SVDLinear.")

        print("\n--- Step 2.5: Verifying Python Pre-Processing Logic ---")
        with torch.no_grad():
            # Reconstruct the full, unquantized weight from its components
            reconstructed_weight_from_svd = (lora_up @ lora_down.T) + residual_weight

            # The forward pass uses smoothed activations, so we must also use the smoothed weight.
            # W_smooth = W * diag(s)
            # The nn.Linear forward pass is y = x @ W.T, so we need (W * diag(s)).T = diag(s) @ W.T
            smoothed_reconstructed_weight = reconstructed_weight_from_svd * s.unsqueeze(0)

            # Perform the forward pass manually in PyTorch
            python_svd_output = input_tensor_for_test @ smoothed_reconstructed_weight.T.to(input_tensor_for_test.dtype)
            if use_bias and bias is not None:
                python_svd_output += bias

            # Calculate the MSE of this ideal, unquantized SVD reconstruction
            svd_reconstruction_mse = torch.mean((original_output - python_svd_output) ** 2)
            print(f"MSE of Python-based SVD reconstruction (pre-quantization): {svd_reconstruction_mse.item():.6f}")
        return svd_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using the fused C++/CUDA kernel.
        """
        # The C++ forward method handles everything:
        # - Fused activation quantization + smoothing + lora_down projection
        # - Fused 4-bit GEMM + lora_up projection + bias addition
        return self.cpp_module.forward(x)

    def __repr__(self):
        return f"SVDLinear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.use_bias})"


# --- Main Test Execution ---
if __name__ == '__main__':
    # 1. Define model parameters
    in_features = 1024
    out_features = 2048
    rank = 32
    batch_size = 32
    device = torch.device("cuda:0")
    dtype = torch.float16

    # 2. Create a standard PyTorch Linear layer to serve as our baseline
    print("--- Step 1: Creating Baseline Layer ---")
    original_linear = nn.Linear(in_features, out_features, bias=True).to(device).to(dtype)
    print("Original torch.nn.Linear layer:\n", original_linear)

    # 3. Create the input tensor *before* calling from_linear
    input_tensor = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    with torch.no_grad():
        original_output = original_linear(input_tensor)

    # 4. Convert the standard layer to our custom SVDLinear layer, passing the test tensor
    print("\n--- Step 2: Converting to SVDLinear ---")
    svd_linear = SVDLinear.from_linear(original_linear, rank=rank, input_tensor_for_test=input_tensor)
    print("\nSuccessfully converted nn.Linear(1024, 2048) to SVDLinear.")
    print("\nConverted SVDLinear layer:\n", svd_linear)


    # 5. Test the REAL C++ forward pass
    print("\n--- Step 3: Testing Forward Pass ---")
    with torch.no_grad():
        svd_output = svd_linear(input_tensor)

    # 6. Compare the outputs
    print(f"\n--- Step 4: Verifying Numerical Correctness ---")
    print("Input tensor shape:", input_tensor.shape)
    print("Original output shape:", original_output.shape)
    print("SVDLinear output shape:", svd_output.shape)

    mse = torch.mean((original_output - svd_output) ** 2)
    print(f"\nMean Squared Error between original and SVDLinear outputs: {mse.item():.6f}")
    
    assert mse.item() < 0.01

    print("\n✅ Verification successful! The SVDLinear layer is working correctly.")