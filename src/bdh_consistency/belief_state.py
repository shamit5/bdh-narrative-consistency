"""Narrative belief state construction, character conditioning, and consistency evaluation."""

from typing import List, Tuple
import torch

from .model import BDH
from .tokenizer import tokenize, make_chunks


def normalize_book_name(name: str) -> str:
    """Standardizes book title string for key lookup."""
    return name.strip().lower()


def build_book_state(
    model: BDH,
    text: str,
    device: str = "cpu",
    chunk_size: int = 512,
    overlap: int = 128,
    cache_path: str = None
) -> torch.Tensor:
    """Processes long-form novel text incrementally to construct a persistent belief state.

    Supports optional disk caching to avoid redundant forward passes over 100k+ word texts.

    Args:
        model: BDH neural model.
        text: Raw text of the full novel.
        device: Computation device ('cuda', 'mps', or 'cpu').
        chunk_size: Maximum sequence length per chunk.
        overlap: Overlap between consecutive sliding chunks.
        cache_path: Optional file path to load/save precomputed state tensor.

    Returns:
        Tensor of shape (1, D) representing the accumulated global narrative state.
    """
    if cache_path:
        from pathlib import Path
        p = Path(cache_path)
        if p.exists():
            return torch.load(p, map_location=device)

    tokens = tokenize(text).to(device)
    chunks = make_chunks(tokens, chunk_size=chunk_size, overlap=overlap)
    state = None

    model.eval()
    with torch.no_grad():
        for ch in chunks:
            if len(ch) < 2:
                continue
            idx = ch[:-1].unsqueeze(0)
            state = model(idx, state=state)[2]

    if state is None:
        state = torch.zeros(1, model.config.n_embd, device=device)

    detached_state = state.detach()
    if cache_path:
        from pathlib import Path
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(detached_state, p)

    return detached_state


def condition_on_character(
    model: BDH,
    book_state: torch.Tensor,
    character: str,
    device: str = "cpu"
) -> torch.Tensor:
    """Adapts global narrative state to focus on constraints specific to a given character.

    Args:
        model: BDH neural model.
        book_state: Precomputed book belief state tensor (1, D).
        character: Name of character to condition on.
        device: Computation device.

    Returns:
        Tensor of shape (1, D) conditioned on character entity constraints.
    """
    prompt = (
        f"The following statements concern the character {character}. "
        f"Only consider facts and history established in the story."
    )
    tokens = tokenize(prompt).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        char_state = model(tokens, state=book_state)[2]
    return char_state.detach()


def evaluate_backstory(
    model: BDH,
    char_state: torch.Tensor,
    backstory: str,
    device: str = "cpu",
    chunk_size: int = 512,
    overlap: int = 128
) -> Tuple[float, List[float]]:
    """Evaluates causal consistency of proposed backstory against character belief state.

    Uses conservative minimum-score aggregation: a single strong contradiction across
    any chunk marks the entire backstory as inconsistent.

    Args:
        model: BDH model.
        char_state: Entity-conditioned belief state tensor (1, D).
        backstory: Text of the proposed character backstory.
        device: Target device.
        chunk_size: Token chunk size.
        overlap: Token chunk overlap.

    Returns:
        Tuple of:
            min_score: Minimum consistency score observed across all chunks.
            scores: List of consistency scores for each chunk.
    """
    tokens = tokenize(backstory).to(device)
    chunks = make_chunks(tokens, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0.0, []

    state = char_state
    scores: List[float] = []

    model.eval()
    with torch.no_grad():
        for ch in chunks:
            if len(ch) < 2:
                continue
            idx = ch[:-1].unsqueeze(0)
            out = model(idx, state=state)
            state = out[2]
            scores.append(out[3].item())

    min_score = min(scores) if scores else 0.0
    return min_score, scores
