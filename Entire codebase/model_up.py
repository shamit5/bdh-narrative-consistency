import dataclasses
import math
import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        # Core sparse projection tensors
        self.encoder = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.encoder_v = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.decoder = nn.Parameter(torch.randn(nh * N, D) * 0.02)

        # Residual gating
        self.gate_alpha = nn.Parameter(torch.tensor(0.0))
        self.gate_beta = nn.Parameter(torch.tensor(0.0))

        self.embed = nn.Embedding(config.vocab_size, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Parameter(torch.randn(D, config.vocab_size) * 0.02)
