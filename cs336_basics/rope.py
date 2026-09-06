import torch.nn
import einops

class RotaryPositionalEmbeddingModule(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float64).float() / d_k))
        pos = torch.arange(max_seq_len, dtype=torch.float64).float()
        # No matching 
        angles = einops.einsum(pos, inv_freq, "seq, half_d -> seq half_d")   # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions]        # (..., seq, d_k/2)
        sin = self.sin[token_positions]        # (..., seq, d_k/2)
        x_pair = einops.rearrange(x, "... (half_d two) -> ... half_d two", two=2)
        x_even = x_pair[..., 0]                # (..., seq, d_k/2)
        x_odd  = x_pair[..., 1]
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        out = torch.stack([out_even, out_odd], dim=-1)   # (..., seq, d_k/2, 2)
        return einops.rearrange(out, "... half_d two -> ... (half_d two)")

