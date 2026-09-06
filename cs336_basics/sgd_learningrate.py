# cs336_basics/sgd.py
from collections.abc import Callable
from typing import Optional
import math
import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})   # base class stores params + defaults

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:        # usually one group; many if per-part lrs
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:             # e.g. frozen/unused params
                    continue
                state = self.state[p]          # per-parameter persistent dict
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * p.grad.data   # eq. (20), in place
                state["t"] = t + 1
        return loss

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr = 1)
for t in range(10):
    opt.zero_grad()
    loss = (weights ** 2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()

weights2 = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt2 = SGD([weights2], lr = 1e2)
for t in range(10):
    opt2.zero_grad()
    loss2 = (weights2 ** 2).mean()
    print("lr2:", loss2.cpu().item())
    loss2.backward()
    opt2.step()


weights3 = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt3 = SGD([weights3], lr = 1e3)
for t in range(10):
    opt3.zero_grad()
    loss3 = (weights3 ** 2).mean()
    print("lr3:", loss3.cpu().item())
    loss3.backward()
    opt3.step()