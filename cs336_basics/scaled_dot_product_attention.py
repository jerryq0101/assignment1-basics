import math

import torch.nn
import einops
from cs336_basics.softmax import SoftmaxModule

class AttentionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # first do default attn, then deal with masks
        # initial impl
        q_m_k = einops.einsum(query, key, "... n d_k, ... m d_k -> ... n m")
        scores = q_m_k / math.sqrt(key.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        sft = SoftmaxModule()
        #  We need the Q * K^T to be in a distribution=1 because we want to get how much of each previous token position to get for this particular token
        sft_res = sft.forward(scores, dim=-1)

        # For this position, how much do we want from all the other positions
        res = sft_res @ value

        return res
