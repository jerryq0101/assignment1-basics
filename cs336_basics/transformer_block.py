import torch.nn
import einops
from cs336_basics.rmsnorm_module import RMSNormModule
from cs336_basics.multihead_attn import MultiheadSelfAttentionModule
from cs336_basics.pos_ffn import SwiGLUModule

class TransformerBlockModule(torch.nn.Module):
    def __init__(self, d_model:int, num_heads: int, d_ff: int, max_seq_len: int, theta: int, eps: int, weights: dict[str, torch.Tensor] = None):
        # Input is gonna be some sort of (batch_size, seq, d_model)
        # Then we do normalization and then self attention
        # Then we do normalization and then positionwise ffn (swiglu)
        # Then we add and get res

        super().__init__()
        self.norm1 = RMSNormModule(d_model, eps=eps)
        self.norm2 = RMSNormModule(d_model, eps=eps)
        self.mha = MultiheadSelfAttentionModule(d_model, num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ffn = SwiGLUModule(d_model, d_ff)

        if weights is not None:
            # Put weights into each of these things
            self.norm1.weight = torch.nn.Parameter(weights["ln1.weight"])
            self.norm2.weight = torch.nn.Parameter(weights["ln2.weight"])
            self.mha.q_proj.weight = torch.nn.Parameter(weights["attn.q_proj.weight"])
            self.mha.k_proj.weight = torch.nn.Parameter(weights["attn.k_proj.weight"])
            self.mha.v_proj.weight = torch.nn.Parameter(weights["attn.v_proj.weight"])
            self.mha.output_proj.weight = torch.nn.Parameter(weights["attn.output_proj.weight"])
            self.ffn.linear_1.weight = torch.nn.Parameter(weights["ffn.w1.weight"])
            self.ffn.linear_2.weight = torch.nn.Parameter(weights["ffn.w2.weight"])
            self.ffn.linear_3.weight = torch.nn.Parameter(weights["ffn.w3.weight"])

    def forward(self, x: torch.Tensor):
        # normed_x = self.norm1.forward(x)
        first_add_pt = x + self.mha.forward_with_rope(x)
        # normed_first_add_pt = self.norm2.forward(first_add_pt)
        second_add_pt = first_add_pt + self.ffn.forward(first_add_pt)
        return second_add_pt

    def norm(self, norm_function, x):
        norm_function(x)
