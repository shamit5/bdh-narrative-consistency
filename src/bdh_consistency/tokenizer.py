"""Byte-level tokenizer and sliding window chunking utilities."""

from typing import List, Union
import torch


def tokenize(text: str) -> torch.Tensor:
    """Encodes text using byte-level tokenization mapping characters to [0, 255].

    Args:
        text: Input string.

    Returns:
        1D torch.LongTensor containing byte values.
    """
    return torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)


def detokenize(tokens: Union[torch.Tensor, List[int]]) -> str:
    """Decodes byte-level tokens back into an ASCII/UTF-8 string representation.

    Args:
        tokens: 1D tensor or list of integers in range [0, 255].

    Returns:
        Decoded string.
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return "".join(chr(t) for t in tokens)


def make_chunks(
    tokens: torch.Tensor,
    chunk_size: int = 512,
    overlap: int = 128
) -> List[torch.Tensor]:
    """Partitions a 1D token tensor into overlapping sliding windows.

    Args:
        tokens: 1D LongTensor of token IDs.
        chunk_size: Maximum sequence length per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of 1D LongTensors, each representing a narrative chunk.
    """
    chunks = []
    i = 0
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(f"chunk_size ({chunk_size}) must be strictly greater than overlap ({overlap})")

    while i < len(tokens) - 1:
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        i += stride

    return chunks
