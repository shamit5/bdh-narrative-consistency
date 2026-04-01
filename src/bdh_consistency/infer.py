"""Interactive CLI inference tool for evaluating character backstories."""

import argparse
from pathlib import Path
import torch

from .config import BDHConfig
from .model import BDH
from .belief_state import normalize_book_name, build_book_state, condition_on_character, evaluate_backstory


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current, current.parent, current.parent.parent]:
        if (p / "Books").exists() and (p / "data").exists():
            return p
    return current.parent.parent


def run_single_inference(
    book_title: str,
    character: str,
    backstory: str,
    weights_path: Path,
    threshold_path: Path,
    books_dir: Path,
    device: str = "cpu"
):
    print("=" * 65)
    print(" 🐉 Baby Dragon Hatchling (BDH) Consistency Reasoner")
    print("=" * 65)
    print(f" Source Novel:    {book_title}")
    print(f" Target Character: {character}")
    print(f" Backstory Length: {len(backstory)} characters")
    print("-" * 65)

    config = BDHConfig(n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=256)
    model = BDH(config).to(device)

    # Load weights or fallback to initialized weights
    has_weights = weights_path.exists() and threshold_path.exists()
    if has_weights:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        threshold = torch.load(threshold_path, map_location=device)["threshold"]
        print(f" Model weights loaded. Calibrated threshold: {threshold:.4f}")
    else:
        threshold = 0.5000
        print(f"⚠️  No trained checkpoint found at {weights_path}.")
        print(f"    Running with initialized weights and default threshold ({threshold:.4f}).")

    model.eval()

    # Find book file
    matched_book = None
    book_norm = normalize_book_name(book_title)
    for p in books_dir.glob("*.txt"):
        if normalize_book_name(p.stem) in book_norm or book_norm in normalize_book_name(p.stem):
            matched_book = p
            break

    if not matched_book:
        print(f"❌ Error: Could not find matching .txt file for '{book_title}' in {books_dir}")
        return

    print(f"📖 Ingesting novel from: {matched_book.name}...")
    with open(matched_book, "r", encoding="utf-8", errors="ignore") as f:
        book_text = f.read()

    book_state = build_book_state(model, book_text, device=device)
    print(" Entity conditioning on character constraints...")
    char_state = condition_on_character(model, book_state, character, device=device)

    print(" Assessing narrative causal compatibility...")
    min_score, scores = evaluate_backstory(model, char_state, backstory, device=device)

    is_consistent = min_score >= threshold
    verdict_emoji = "✅" if is_consistent else "❌"
    verdict_text = "CONSISTENT" if is_consistent else "INCONSISTENT (Contradiction Detected)"

    print("\n" + "=" * 65)
    print(f" VERDICT: {verdict_emoji} {verdict_text}")
    print("=" * 65)
    print(f" Minimum Chunk Consistency Score: {min_score:.4f}")
    print(f" Decision Threshold:              {threshold:.4f}")
    print(f" Evaluated Chunks:                {len(scores)}")

    if scores:
        print("\n Chunk Consistency Profile:")
        for idx, sc in enumerate(scores):
            bar_len = int(sc * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            status = "PASS" if sc >= threshold else "FAIL"
            print(f"   Chunk {idx+1:02d}: [{bar}] {sc:.3f} ({status})")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a character backstory for narrative consistency.")
    project_root = find_project_root()
    parser.add_argument("--book", type=str, default="The Count of Monte Cristo", help="Novel title")
    parser.add_argument("--char", type=str, default="Noirtier", help="Character name")
    parser.add_argument(
        "--backstory",
        type=str,
        default="Learning that Villefort meant to denounce him to Louis XVIII, Noirtier pre-emptively handed the conspiracy dossier to a British spy.",
        help="Proposed backstory to test",
    )
    parser.add_argument("--weights-path", type=Path, default=project_root / "bdh_trained.pt")
    parser.add_argument("--threshold-path", type=Path, default=project_root / "threshold.pt")
    parser.add_argument("--books-dir", type=Path, default=project_root / "Books")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    run_single_inference(
        book_title=args.book,
        character=args.char,
        backstory=args.backstory,
        weights_path=args.weights_path,
        threshold_path=args.threshold_path,
        books_dir=args.books_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
