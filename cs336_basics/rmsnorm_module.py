import torch.nn
import einops

class RMSNormModule(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype =None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, sequence_length, d_model)
        # so for each vector I should have a number for every activation vector
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # compute the mean square of the activations
        mean_square = einops.reduce(x ** 2, "... d_model -> ... 1", "mean")
        mean_square = mean_square + self.eps
        # Compute root mean square
        rms = torch.sqrt(mean_square)
        # Normalize the activations
        x_normalized = x / rms * self.weight
        # Convert back to original dtype
        x_normalized = x_normalized.to(in_dtype)
        return x_normalized
