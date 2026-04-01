"""Unit tests for byte tokenizer and chunking utilities."""

import pytest
import torch
from bdh_consistency.tokenizer import tokenize, detokenize, make_chunks


def test_tokenize_range():
    text = "The Count of Monte Cristo: Revenge and Redemption! 1844."
    tokens = tokenize(text)
    assert isinstance(tokens, torch.Tensor)
    assert len(tokens) == len(text)
    assert (tokens >= 0).all() and (tokens < 256).all()


def test_detokenize_roundtrip():
    original = "Hello World! Baby Dragon Hatchling consistency."
    tokens = tokenize(original)
    recovered = detokenize(tokens)
    assert recovered == original


def test_make_chunks_partitioning():
    tokens = torch.arange(1000)
    chunk_size = 512
    overlap = 128
    stride = chunk_size - overlap

    chunks = make_chunks(tokens, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) > 1
    assert len(chunks[0]) == chunk_size
    # Second chunk start must equal stride
    assert chunks[1][0].item() == stride


def test_make_chunks_invalid_args():
    tokens = torch.arange(100)
    with pytest.raises(ValueError):
        make_chunks(tokens, chunk_size=100, overlap=100)
