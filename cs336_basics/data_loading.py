import torch.nn
import numpy as np

class DataLoadingModule():
    def __init__(self):
        pass

    def get_batch(self, x: np.ndarray, batch_size: int, context_length: int, device: str):
        max_start = len(x) - context_length          # need s + context_length + 1 <= len(x)
        starts = np.random.randint(0, max_start, size=batch_size)
        inputs  = np.stack([x[s     : s + context_length]     for s in starts])
        targets = np.stack([x[s + 1 : s + context_length + 1] for s in starts])
        return (
            torch.from_numpy(inputs.astype(np.int64)).to(device),
            torch.from_numpy(targets.astype(np.int64)).to(device),
        )
