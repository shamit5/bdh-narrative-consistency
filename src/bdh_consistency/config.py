"""Configuration dataclass for Baby Dragon Hatchling (BDH) architecture."""

from dataclasses import dataclass


@dataclass
class BDHConfig:
    """Hyperparameters for the BDH neural consistency network.

    Attributes:
        n_layer: Number of BDH reasoning blocks.
        n_embd: Embedding and narrative state dimensionality D.
        dropout: Dropout rate applied across residual activations.
        n_head: Number of attention heads (nh).
        mlp_internal_dim_multiplier: Scaling factor for sparse latent dimension N = multiplier * D // nh.
        vocab_size: Byte-level vocabulary size (256 for standard byte encoding).
    """
    n_layer: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 8
    vocab_size: int = 256
