import torch.nn
import einops
from cs336_basics.softmax import SoftmaxModule

class CrossEntropyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, targets):
        m = x.amax(dim=-1, keepdim=True)
        lse = m.squeeze(-1) + torch.log(torch.exp(x - m).sum(dim=-1))
        target_logit = x.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (lse - target_logit).mean()
    