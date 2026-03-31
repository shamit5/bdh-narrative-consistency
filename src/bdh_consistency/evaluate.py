"""Inference and benchmark evaluation script for BDH consistency model."""

import argparse
from pathlib import Path
import pandas as pd
import torch

from .config import BDHConfig
from .model import BDH
from .tokenizer import tokenize, make_chunks
from .belief_state import normalize_book_name, build_book_state, condition_on_character


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current, current.parent, current.parent.parent]:
        if (p / "Books").exists() and (p / "data").exists():
            return p
    return current.parent.parent


def evaluate(
    test_path: Path,
    books_dir: Path,
    weights_path: Path,
    threshold_path: Path,
    output_path: Path,
    device: str = "auto"
):
    if device == "auto":
        device = get_default_device()

    print(f"🔧 Device: {device}")
    config = BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        vocab_size=256
    )

    model = BDH(config).to(device)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Train the model first.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found at {threshold_path}.")
    threshold = torch.load(threshold_path, map_location=device)["threshold"]
    print(f" Loaded model checkpoint & decision threshold: {threshold:.4f}")

    df = pd.read_csv(test_path)
    df = df[df["content"].notna()].reset_index(drop=True)
    print(f" Loaded {len(df)} test samples from {test_path}")

    book_paths = {
        "The Count of Monte Cristo": books_dir / "The Count of Monte Cristo.txt",
        "In Search of the Castaways": books_dir / "In search of the castaways.txt",
    }

    print("📚 Building book narrative states...")
    book_states = {}
    for name, path in book_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Book file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            book_states[normalize_book_name(name)] = build_book_state(model, f.read(), device=device)

    print(" Evaluating backstories using minimum-score consistency rule...")
    results = []

    with torch.no_grad():
        for row in df.itertuples():
            book_key = normalize_book_name(row.book_name)
            if book_key not in book_states:
                results.append([row.id, 0, "missing_book"])
                continue

            book_state = book_states[book_key]
            char_state = condition_on_character(model, book_state, row.char, device=device)

            tokens = tokenize(row.content).to(device)
            chunks = make_chunks(tokens)
            if not chunks:
                results.append([row.id, 0, "empty_content"])
                continue

            state = char_state
            scores = []

            for ch in chunks:
                if len(ch) < 2:
                    continue
                out = model(ch[:-1].unsqueeze(0), state=state)
                state = out[2]
                scores.append(out[3].item())

            final_score = min(scores) if scores else 0.0
            pred = int(final_score >= threshold)
            results.append([row.id, pred, f"min_score={final_score:.3f}"])

    out_df = pd.DataFrame(results, columns=["id", "prediction", "rationale"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"✅ Evaluation complete. Saved {len(out_df)} predictions to {output_path}")

    consistent_count = (out_df["prediction"] == 1).sum()
    contradict_count = (out_df["prediction"] == 0).sum()
    print(f"📊 Summary: Consistent: {consistent_count} | Contradict: {contradict_count}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDH consistency model.")
    project_root = find_project_root()
    parser.add_argument("--test-path", type=Path, default=project_root / "data" / "test.csv")
    parser.add_argument("--books-dir", type=Path, default=project_root / "Books")
    parser.add_argument("--weights-path", type=Path, default=project_root / "bdh_trained.pt")
    parser.add_argument("--threshold-path", type=Path, default=project_root / "threshold.pt")
    parser.add_argument("--output-path", type=Path, default=project_root / "results.csv")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    evaluate(
        test_path=args.test_path,
        books_dir=args.books_dir,
        weights_path=args.weights_path,
        threshold_path=args.threshold_path,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
