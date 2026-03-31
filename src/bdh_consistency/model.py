"""Baby Dragon Hatchling (BDH) Neural Architecture for Long-Context Consistency Reasoning."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BDHConfig


class BDH(nn.Module):
    """Baby Dragon Hatchling (BDH) model with persistent internal belief state.

    The model maps token sequences into a sparse latent representation,
    conditions reasoning on an evolving narrative belief state, and evaluates
    consistency scores per chunk against accumulated constraints.
    """

    def __init__(self, config: Optional[BDHConfig] = None):
        super().__init__()
        self.config = config or BDHConfig()

        nh = self.config.n_head
        D = self.config.n_embd
        N = self.config.mlp_internal_dim_multiplier * D // nh

        # Sparse projection core tensors
        self.encoder = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.encoder_v = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.decoder = nn.Parameter(torch.randn(nh * N, D) * 0.02)

        # Residual gating parameters
        self.gate_alpha = nn.Parameter(torch.tensor(0.0))
        self.gate_beta = nn.Parameter(torch.tensor(0.0))

        # Narrative belief state projection & gating
        self.state_proj = nn.Linear(D, D, bias=False)
        self.state_gate = nn.Linear(2 * D, D, bias=False)
        self.consistency_proj = nn.Linear(D, 1)

        # State-mediated attention keys & values
        self.state_key = nn.Linear(D, D, bias=False)
        self.state_value = nn.Linear(D, D, bias=False)

        # Project sparse latent dimension N back to embedding dimension D
        self.latent_to_D = nn.Linear(N, D, bias=False)

        # Token embedding and norm
        self.embed = nn.Embedding(self.config.vocab_size, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        self.drop = nn.Dropout(self.config.dropout)
        self.lm_head = nn.Parameter(torch.randn(D, self.config.vocab_size) * 0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass over token sequence idx conditioned on state.

        Args:
            idx: LongTensor of shape (B, T) with token IDs in [0, vocab_size - 1].
            targets: Optional LongTensor of shape (B, T) for language modeling cross-entropy.
            state: Optional FloatTensor of shape (B, D) representing narrative belief state.

        Returns:
            Tuple of:
                logits: (B, T, vocab_size) next-token logits
                loss: scalar cross-entropy loss if targets provided, else None
                new_state: (B, D) updated narrative belief state
                consistency_score: (B, 1) consistency score in [0, 1]
                surprise: (B, 1) magnitude of representation shift
        """
        B, T = idx.size()
        D = self.config.n_embd

        # Initialize empty state if not provided
        if state is None:
            state = torch.zeros(B, D, device=idx.device)

        # Embed tokens and fuse with narrative belief state
        x = self.embed(idx)
        x = x + state.unsqueeze(1)
        x = self.ln(x).unsqueeze(1)  # (B, 1, T, D)

        # Multi-layer BDH reasoning blocks
        for _ in range(self.config.n_layer):
            # Sparse latent projection
            x_latent = x @ self.encoder  # (B, nh, T, N)
            x_sparse = F.relu(x_latent)

            # Alignment with narrative belief state
            x_exp = self.latent_to_D(x_sparse)  # (B, nh, T, D)
            state_k = self.state_key(state).unsqueeze(1).unsqueeze(2).expand(-1, self.config.n_head, -1, -1)
            state_v = self.state_value(state).unsqueeze(1).unsqueeze(2).expand(-1, self.config.n_head, -1, -1)

            alignment = (x_exp * state_k).sum(dim=-1, keepdim=True)
            yKV = self.ln(alignment * state_v)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)

            # Residual multiplicative blending
            alpha = torch.sigmoid(self.gate_alpha)
            beta = torch.sigmoid(self.gate_beta)
            xy_sparse = alpha * (x_sparse * y_sparse) + beta * (x_sparse + y_sparse)
            xy_sparse = self.drop(xy_sparse)

            # Decode back to embedding space
            y = (
                xy_sparse.transpose(1, 2)
                .reshape(B, 1, T, -1)
                @ self.decoder
            )
            x = self.ln(x + y)

        # Output LM logits
        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Narrative state update dynamics
        chunk_summary = x.mean(dim=2).squeeze(1)  # (B, D)
        surprise = torch.norm(chunk_summary - state, dim=-1, keepdim=True)
        update_gate = torch.sigmoid(self.state_gate(torch.cat([state, chunk_summary], dim=-1)))

        # If previous state was empty, immediately absorb chunk
        is_empty = (state.abs().sum(dim=-1, keepdim=True) < 1e-6).float()
        controlled_gate = update_gate * torch.exp(-surprise) + is_empty
        controlled_gate = controlled_gate.clamp(0.0, 1.0)

        proposed_state = self.state_proj(chunk_summary)
        new_state = controlled_gate * state + (1.0 - controlled_gate) * proposed_state

        # Consistency score via deviation projection
        consistency_score = torch.sigmoid(self.consistency_proj(chunk_summary - state))

        return logits, loss, new_state, consistency_score, surprise

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        state: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive text generation conditioned on narrative belief state."""
        for _ in range(max_new_tokens):
            logits, _, state, _, _ = self(idx, state=state)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx, state
