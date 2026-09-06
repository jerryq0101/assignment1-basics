import torch.nn
import einops

class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # x is of shape (..., d_model)
        # we want to apply softmax along the specified dimension
        x_max = x.amax(dim=dim, keepdim=True)
        x_subed = x - x_max
        x_exp = torch.exp(x_subed)
        x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
        return x_exp / x_exp_sum
