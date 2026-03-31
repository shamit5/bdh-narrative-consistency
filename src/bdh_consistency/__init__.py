"""Baby Dragon Hatchling (BDH) Consistency Reasoning Package."""

from .config import BDHConfig
from .model import BDH
from .tokenizer import tokenize, detokenize, make_chunks
from .belief_state import (
    normalize_book_name,
    build_book_state,
    condition_on_character,
    evaluate_backstory,
)

__version__ = "1.0.0"

__all__ = [
    "BDHConfig",
    "BDH",
    "tokenize",
    "detokenize",
    "make_chunks",
    "normalize_book_name",
    "build_book_state",
    "condition_on_character",
    "evaluate_backstory",
]
