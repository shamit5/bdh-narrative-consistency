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

        self.encoder = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.encoder_v = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.decoder = nn.Parameter(torch.randn(nh * N, D) * 0.02)

        self.gate_alpha = nn.Parameter(torch.tensor(0.0))
        self.gate_beta = nn.Parameter(torch.tensor(0.0))

        # Narrative state projection
        self.state_proj = nn.Linear(D, D, bias=False)
        self.state_gate = nn.Linear(2 * D, D, bias=False)
        self.state_key = nn.Linear(D, D, bias=False)
        self.state_value = nn.Linear(D, D, bias=False)
        self.latent_to_D = nn.Linear(N, D, bias=False)

        self.embed = nn.Embedding(config.vocab_size, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Parameter(torch.randn(D, config.vocab_size) * 0.02)

    def forward(self, idx, targets=None, state=None):
        B, T = idx.size()
        D = self.config.n_embd
        if state is None:
            state = torch.zeros(B, D, device=idx.device)
        x = self.embed(idx) + state.unsqueeze(1)
        x = self.ln(x).unsqueeze(1)
        for _ in range(self.config.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            x_exp = self.latent_to_D(x_sparse)
            state_k = self.state_key(state).unsqueeze(1).unsqueeze(2).expand(-1, self.config.n_head, -1, -1)
            state_v = self.state_value(state).unsqueeze(1).unsqueeze(2).expand(-1, self.config.n_head, -1, -1)
            alignment = (x_exp * state_k).sum(dim=-1, keepdim=True)
            yKV = self.ln(alignment * state_v)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            alpha = torch.sigmoid(self.gate_alpha)
            beta = torch.sigmoid(self.gate_beta)
            xy_sparse = alpha * (x_sparse * y_sparse) + beta * (x_sparse + y_sparse)
            xy_sparse = self.drop(xy_sparse)
            y = xy_sparse.transpose(1, 2).reshape(B, 1, T, -1) @ self.decoder
            x = self.ln(x + y)
        chunk_summary = x.mean(dim=2).squeeze(1)
        surprise = torch.norm(chunk_summary - state, dim=-1, keepdim=True)
        update_gate = torch.sigmoid(self.state_gate(torch.cat([state, chunk_summary], dim=-1)))
        is_empty = (state.abs().sum(dim=-1, keepdim=True) < 1e-6)
        controlled_gate = (update_gate * torch.exp(-surprise) + is_empty).clamp(0, 1)
        proposed_state = self.state_proj(chunk_summary)
        new_state = controlled_gate * state + (1 - controlled_gate) * proposed_state
        return x, None, new_state, surprise
