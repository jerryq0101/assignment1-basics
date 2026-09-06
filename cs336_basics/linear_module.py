import math

import torch.nn
from einops import einsum

class LinearModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # initialize weights from a bell curve, with variance 2/(d_in + d_out)
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, in_features)
        # weight is of shape (out_features, in_features)
        # Therefore weight should be transposed to (in_features, out_features
        # Use einsum to perform the matrix multiplication
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
