import torch.nn
import einops
from cs336_basics.transformer_block import TransformerBlockModule
from cs336_basics.embedding_module import EmbeddingModule
from cs336_basics.rmsnorm_module import RMSNormModule
from cs336_basics.linear_module import LinearModule

class TransformerLMModule(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                 rope_theta, weights=None, eps=1e-5, device=None, dtype=None):
        super().__init__()
        if weights is not None:
            device = weights["ln_final.weight"].device
            dtype = weights["ln_final.weight"].dtype

        self.token_embeddings = EmbeddingModule(vocab_size, d_model, device=device, dtype=dtype)
        if weights is not None:
            self.token_embeddings.weight = torch.nn.Parameter(weights["token_embeddings.weight"])

        blocks = []
        for i in range(num_layers):
            layer_weights = None
            if weights is not None:
                prefix = f"layers.{i}."
                layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            blocks.append(TransformerBlockModule(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff, weights=layer_weights,
                max_seq_len=context_length, theta=rope_theta, eps=eps,
            ))
        self.layers = torch.nn.ModuleList(blocks)

        self.ln_final = RMSNormModule(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = LinearModule(d_model, vocab_size, device=device, dtype=dtype)
        if weights is not None:
            self.ln_final.weight = torch.nn.Parameter(weights["ln_final.weight"])
            self.lm_head.weight = torch.nn.Parameter(weights["lm_head.weight"])


    def forward(self, x: torch.Tensor):
        # so this tensor will have batch_size and sequence length
        # Convert through embeddings
        x = self.token_embeddings.forward(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x
    