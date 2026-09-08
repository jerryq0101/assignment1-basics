import torch.nn
import einops
from cs336_basics.linear_module import LinearModule

class SwiGLUModule(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # linear 1, linear 2 are d ff x dmodel
        # linear 2 is d model times dff
        self.linear_1 = LinearModule(d_model, d_ff, device=device, dtype=dtype)
        self.linear_3 = LinearModule(d_model, d_ff, device=device, dtype=dtype)
        self.linear_2 = LinearModule(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (..., d_model), we just need to act on the d_model
        Wx_1 = self.linear_1(x)
        Wx_3 = self.linear_3(x)
        SiLU_Wx_1 = Wx_1 * torch.sigmoid(Wx_1)
        return self.linear_2(SiLU_Wx_1 * Wx_3)

class SiLUModule(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear_1 = LinearModule(d_model, d_ff, device=device, dtype=dtype)
        self.linear_2 = LinearModule(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx_1 = self.linear_1(x)
        SiLU_Wx_1 = Wx_1 * torch.sigmoid(Wx_1)
        return self.linear_2(SiLU_Wx_1)
