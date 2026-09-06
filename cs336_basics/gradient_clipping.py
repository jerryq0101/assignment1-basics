import torch.nn
import math

class GradientClipping:
    def __init__(self):
        pass

    def clip(self, parameters, max_l2_norm):
        grads = [p.grad for p in parameters if p.grad is not None]
        total_sum_g2 = sum((g**2).sum() for g in grads)
        sqrt_of = torch.sqrt(total_sum_g2)
        if sqrt_of <= max_l2_norm:
            return parameters
        else:
            scale = max_l2_norm / (sqrt_of + 1e-6)
            for g in grads:
                g.mul_(scale)
            return parameters
