import torch.nn
import numpy as np


class CheckpointingModule:
    def __init__(self):
        pass

    def save_checkpoint(self, model, optimizer, iteration, hyper, training_time, out):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "hyperparams": hyper,
                "training_time_used": training_time,
            },
            out,
        )

    def load_checkpoint(self, src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        item = torch.load(src, map_location="cpu")  # in case loading on different device that doens't have mps
        modeldict = item["model"]
        optimizerdict = item["optimizer"]
        model.load_state_dict(modeldict)
        optimizer.load_state_dict(optimizerdict)
        
        return (item["iteration"], item["hyperparams"], item["training_time_used"])
