"""Unit tests for narrative belief state building and character conditioning."""

import torch
from bdh_consistency.config import BDHConfig
from bdh_consistency.model import BDH
from bdh_consistency.belief_state import (
    normalize_book_name,
    condition_on_character,
    evaluate_backstory,
)


def test_normalize_book_name():
    assert normalize_book_name("  The Count of Monte Cristo \n") == "the count of monte cristo"
    assert normalize_book_name("In Search of the Castaways") == "in search of the castaways"


def test_condition_on_character():
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=256)
    model = BDH(config)
    book_state = torch.randn(1, 64)

    char_state = condition_on_character(model, book_state, "Noirtier", device="cpu")
    assert char_state.shape == (1, 64)
    assert not torch.allclose(char_state, book_state)


def test_evaluate_backstory():
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=256)
    model = BDH(config)
    char_state = torch.randn(1, 64)
    backstory = "He spent ten years in quiet exile preparing his secret documents."

    min_score, scores = evaluate_backstory(model, char_state, backstory, device="cpu")
    assert isinstance(min_score, float)
    assert 0.0 <= min_score <= 1.0
    assert len(scores) >= 1
