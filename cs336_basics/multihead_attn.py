import torch.nn
import einops
from cs336_basics.linear_module import LinearModule
from cs336_basics.scaled_dot_product_attention import AttentionModule
from cs336_basics.rope import RotaryPositionalEmbeddingModule

class MultiheadSelfAttentionModule(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        # Initialize multihead self attention stuff
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # Initialize the attention
        self.attn = AttentionModule()

        # Initialize linear layers for Q, K, V 
        # So we have the total sized Q K V, then we would divided the d_model into num_heads 
        # Then, we call Attention on each of the individual heads, then concatenate the results
        # Then, pass through final linear layer
        self.q_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)
        # Initialize the role
        self.rope = RotaryPositionalEmbeddingModule(theta, self.d_head, max_seq_len, device=device)


    def forward_with_rope(self, x: torch.Tensor, token_positions=None):
        # Find the Q K V for each one
        # rearrange for Q K V to be split into different stuff
        # Process through attention individually
        # Concat back together
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        # Reorganize into head dimension stuff 
        q = einops.rearrange(q, "... seq (head d_head) -> ... head seq d_head", head = self.num_heads)
        k = einops.rearrange(k, "... seq (head d_head) -> ... head seq d_head", head = self.num_heads)
        v = einops.rearrange(v, "... seq (head d_head) -> ... head seq d_head", head = self.num_heads)

        if token_positions is None:
             token_positions = torch.arange(x.shape[-2], device=x.device)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        # need to call self.attn on each head dimension
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = self.attn.forward(q, k, v, mask)

        # Compress now
        concated_out = einops.rearrange(out, "... heads seq d_h -> ... seq (heads d_h)", heads=self.num_heads)
        res = self.output_proj.forward(concated_out)
        return res
        
    def forward(self, x: torch.Tensor, token_positions=None):
            # Find the Q K V for each one
            # rearrange for Q K V to be split into different stuff
            # Process through attention individually
            # Concat back together
            q = self.q_proj.forward(x)
            k = self.k_proj.forward(x)
            v = self.v_proj.forward(x)
            # Reorganize into head dimension stuff 
            q = einops.rearrange(q, "... seq (head d_head) -> ... head seq d_head", head = self.num_heads)
            k = einops.rearrange(k, "... seq (head d_head) -> ... head seq d_head", head = self.num_heads)
            v = einops.rearrange(v, "... seq (head d_head) -> ... head seq d_head", head = self.num_heads)
    
            # need to call self.attn on each head dimension
            seq_len = x.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            out = self.attn.forward(q, k, v, mask)
    
            # Compress now
            concated_out = einops.rearrange(out, "... heads seq d_h -> ... seq (heads d_h)", heads=self.num_heads)
            res = self.output_proj.forward(concated_out)
            return res