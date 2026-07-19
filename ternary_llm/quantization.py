import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TernaryQuantizer(torch.autograd.Function):
    """Ternary quantization {-1, 0, +1} with Straight-Through Estimator.

    Absmean quantization: threshold γ = mean(|W|)
    Forward:  clamp(W/γ, -1, 1) → round → {-1, 0, +1}
    Backward: STE passes grad straight through
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        # Absmean threshold (fused clamp avoids conditional branch + tensor alloc)
        gamma = input.abs().mean().clamp(min=1e-6)

        # Normalize, clamp to [-1, 1], then round to ternary
        w_ternary = (input / gamma).clamp(-1, 1).round()

        ctx.save_for_backward(input, gamma)
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: pass gradient straight through
        return grad_output


class Int8Quantizer(torch.autograd.Function):
    """Symmetric INT8 quantization for activations (fake-quantize).

    Forward:  scale = max(|x|) / 127, quantize to [-128, 127], dequantize back to float
    Backward: STE passes grad through as-is

    Output stays float to maintain gradient flow during training.
    Real INT8 conversion happens at inference time.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        scale = input.abs().max() / 127.0
        scale = scale.clamp(min=1e-6)

        # Fake quantize: quantize then dequantize (keeps float, simulates INT8)
        q_dequant = (input / scale).round().clamp(-128, 127) * scale

        ctx.save_for_backward(input, scale)
        return q_dequant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: pass gradient through as-is
        return grad_output


def ternary_quantize(weights: torch.Tensor) -> torch.Tensor:
    """Apply ternary quantization to a weight tensor (inference)."""
    return TernaryQuantizer.apply(weights)


def int8_quantize(activations: torch.Tensor) -> torch.Tensor:
    """Apply INT8 quantization to activation tensor."""
    return Int8Quantizer.apply(activations)


def pack_ternary(weights: torch.Tensor) -> bytes:
    """Pack ternary weights {-1, 0, +1} to 2-bit format (4 weights/byte).

    Encoding: 00=-1, 01=0, 10=+1, MSB first.
    Input: flat float tensor with values in {-1, 0, +1}.
    Output: packed bytes.
    """
    w = weights.detach().cpu().flatten().to(torch.int8)
    # Map {-1,0,+1} -> {0,1,2}
    w = (w + 1).to(torch.uint8)  # -1->0, 0->1, +1->2
    n = len(w)
    # Pad to multiple of 4
    padded = (n + 3) // 4 * 4
    if padded != n:
        w = torch.nn.functional.pad(w, (0, padded - n), value=1)  # pad with 0 (encoded as 1)
    w = w.view(-1, 4)
    # Pack: MSB first, 2 bits per weight
    packed = (w[:, 0].to(torch.uint8) << 6 |
              w[:, 1].to(torch.uint8) << 4 |
              w[:, 2].to(torch.uint8) << 2 |
              w[:, 3].to(torch.uint8))
    return bytes(packed.tolist())


def unpack_ternary(packed: bytes, shape: tuple, device: str = "cpu") -> torch.Tensor:
    """Unpack 2-bit packed ternary weights back to {-1, 0, +1} float tensor.

    Encoding: 00=-1, 01=0, 10=+1, MSB first.
    """
    import numpy as np
    data = np.frombuffer(packed, dtype=np.uint8)
    # Unpack 4 weights per byte, MSB first
    w0 = ((data >> 6) & 3).astype(np.int8) - 1
    w1 = ((data >> 4) & 3).astype(np.int8) - 1
    w2 = ((data >> 2) & 3).astype(np.int8) - 1
    w3 = ((data >> 0) & 3).astype(np.int8) - 1
    flat = np.stack([w0, w1, w2, w3], axis=-1).flatten()
    flat = flat[:np.prod(shape)]
    return torch.from_numpy(flat.astype(np.float32)).reshape(shape).to(device)
