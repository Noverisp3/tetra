import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, StochasticTernaryLinear


class TernarySSMBlock(nn.Module):
    """Ternary State Space Model block (simplified Mamba).

    Architecture:
        x → RMSNorm → TernaryLinear(expand 2×) → split x, gate
        x → Conv1d → SiLU → SSM scan
        gate → SiLU → multiply → TernaryLinear(project back)

    SSM recurrence (diagonal, per-channel):
        h_t = (1 - Δ·A) ⊙ h_{t-1} + Δ·B·x_t
        y_t = h_t  (identity observation, C=I)

    All projections use ternary weights.
    Recurrence is element-wise (d_state = inner_dim for simplicity).

    Args:
        hidden_dim: model dimension
        expand_factor: expansion factor for inner dim (default: 2)
        scale: ternary weight scale
        threshold: bit-flip threshold
        int8: use INT8 forward
    """

    def __init__(self, hidden_dim, expand_factor=2,
                 scale=1.0, threshold=None, int8=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        inner_dim = hidden_dim * expand_factor

        self.norm = RMSNorm(hidden_dim)

        # Fused input projection: hidden_dim → 2 × inner_dim (x, gate)
        self.x_proj = StochasticTernaryLinear(
            hidden_dim, 2 * inner_dim, scale=scale, threshold=threshold, int8=int8
        )

        # Conv1d for local context (depthwise, same-length)
        self.conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim
        )

        # SSM parameters (per-channel)
        self.log_delta = nn.Parameter(torch.zeros(inner_dim))
        self.A_log = nn.Parameter(torch.zeros(inner_dim))

        # Output projection
        self.out_proj = StochasticTernaryLinear(
            inner_dim, hidden_dim, scale=scale, threshold=threshold, int8=int8
        )

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)

        # Fused projection → split
        fused = self.x_proj(x)  # (B, T, 2×inner)
        x_in, gate = fused.chunk(2, dim=-1)

        # Conv1d: (B, T, C) → (B, C, T) → Conv → (B, C, T) → (B, T, C)
        x_conv = self.conv(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM scan (per-channel, diagonal, stable recurrence)
        delta = F.softplus(self.log_delta)  # (inner_dim,)
        # A_gate = sigmoid(A) ∈ (0, 1), so (1 - Δ·A_gate) ∈ (0, 1)
        A_gate = torch.sigmoid(self.A_log)  # (inner_dim,)

        bsize, seq_len, inner_dim = x_conv.shape
        h = torch.zeros(bsize, inner_dim, device=x.device)
        y = torch.empty(bsize, seq_len, inner_dim, device=x.device)
        for t in range(seq_len):
            # h_t = (1 - Δ·A_gate) ⊙ h_{t-1} + Δ·x_t
            decay = 1 - delta * A_gate
            h = decay * h + delta * x_conv[:, t]
            y[:, t] = h

        # Gate: y × SiLU(gate)
        out = y * F.silu(gate)
        return self.out_proj(out) + residual

    @torch.no_grad()
    def apply_bit_flips(self):
        self.x_proj.apply_bit_flips()
        self.out_proj.apply_bit_flips()
