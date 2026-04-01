"""Unit tests for BDH model architecture and tensor mechanics."""

import pytest
import torch
from bdh_consistency.config import BDHConfig
from bdh_consistency.model import BDH


def test_model_initialization():
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=256)
    model = BDH(config)
    assert model is not None
    assert model.encoder.shape == (2, 64, 4 * 64 // 2)


def test_model_forward_pass_shapes():
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=256)
    model = BDH(config)
    B, T = 2, 16
    idx = torch.randint(0, 256, (B, T))

    logits, loss, new_state, score, surprise = model(idx)

    assert logits.shape == (B, T, 256)
    assert loss is None
    assert new_state.shape == (B, 64)
    assert score.shape == (B, 1)
    assert surprise.shape == (B, 1)


def test_model_loss_computation():
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=256)
    model = BDH(config)
    B, T = 2, 16
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))

    logits, loss, _, _, _ = model(idx, targets=targets)
    assert loss is not None
    assert loss.item() > 0.0


def test_consistency_score_bounds():
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=256)
    model = BDH(config)
    idx = torch.randint(0, 256, (1, 32))
    state = torch.randn(1, 64)

    _, _, _, score, _ = model(idx, state=state)
    val = score.item()
    assert 0.0 <= val <= 1.0


def test_backward_gradient_flow():
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=256)
    model = BDH(config)
    idx = torch.randint(0, 256, (1, 16))
    _, _, new_state, score, _ = model(idx)

    # Loss combining consistency score and updated state
    loss = score.sum() + new_state.sum()
    loss.backward()

    assert model.encoder.grad is not None
    assert model.consistency_proj.weight.grad is not None
    assert model.state_gate.weight.grad is not None
    assert model.state_proj.weight.grad is not None
