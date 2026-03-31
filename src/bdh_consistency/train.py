"""Training pipeline for Baby Dragon Hatchling consistency reasoning network."""

import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

from .config import BDHConfig
from .model import BDH
from .tokenizer import tokenize, make_chunks
from .belief_state import normalize_book_name, build_book_state, condition_on_character


def get_default_device() -> str:
    """Detects available hardware accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_project_root() -> Path:
    """Finds project root directory by inspecting markers."""
    current = Path(__file__).resolve().parent
    for p in [current, current.parent, current.parent.parent]:
        if (p / "Books").exists() and (p / "data").exists():
            return p
    return current.parent.parent


def train(
    data_path: Path,
    books_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    lr: float = 3e-4,
    device: str = "auto"
):
    """Executes multi-epoch consistency training loop."""
    if device == "auto":
        device = get_default_device()

    print(f"🔧 Using device: {device}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        vocab_size=256
    )

    model = BDH(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Load dataset
    print(f"📖 Reading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    df["label"] = df["label"].str.strip().str.lower().map({"consistent": 1, "contradict": 0})
    assert df["label"].isna().sum() == 0, "Labels must be either 'consistent' or 'contradict'"

    df = df[df["content"].notna()].reset_index(drop=True)
    train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    print(f" Loaded {len(train_df)} training samples.")

    # Identify books
    book_paths = {
        "The Count of Monte Cristo": books_dir / "The Count of Monte Cristo.txt",
        "In Search of the Castaways": books_dir / "In search of the castaways.txt",
    }

    print("📚 Precomputing persistent book narrative belief states...")
    book_states = {}
    for name, path in book_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Book file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            book_states[normalize_book_name(name)] = build_book_state(model, f.read(), device=device)
        print(f"  ✓ Ingested '{name}' state.")

    print(f"\n🚀 Commencing training for {epochs} epochs...")
    epoch_thresholds = []

    for ep in range(epochs):
        model.train()
        pos_scores, neg_scores = [], []
        total_loss = 0.0

        for row in train_df.itertuples():
            book_name_norm = normalize_book_name(row.book_name)
            if book_name_norm not in book_states:
                continue

            book_state = book_states[book_name_norm]
            char_state = condition_on_character(model, book_state, row.char, device=device)

            tokens = tokenize(row.content).to(device)
            chunks = make_chunks(tokens)
            if not chunks:
                continue

            state = char_state
            scores = []

            for ch in chunks:
                if len(ch) < 2:
                    continue
                out = model(ch[:-1].unsqueeze(0), state=state)
                state = out[2]
                scores.append(out[3].squeeze())

            if not scores:
                continue

            final_score = torch.stack(scores).min()
            label_tensor = torch.tensor([row.label], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy(final_score.view(1), label_tensor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            (pos_scores if row.label == 1 else neg_scores).append(final_score.detach().item())

        scheduler.step()

        mean_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
        mean_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
        threshold = (mean_pos + mean_neg) / 2 if (pos_scores and neg_scores) else 0.5
        epoch_thresholds.append(threshold)

        print(
            f"Epoch {ep+1:02d}/{epochs:02d} | Loss: {total_loss:6.3f} | "
            f"Consistent Avg: {mean_pos:5.3f} | Contradict Avg: {mean_neg:5.3f} | Threshold: {threshold:5.3f}"
        )

    final_threshold = sum(epoch_thresholds[-5:]) / 5 if len(epoch_thresholds) >= 5 else epoch_thresholds[-1]
    model_save_path = output_dir / "bdh_trained.pt"
    threshold_save_path = output_dir / "threshold.pt"

    torch.save(model.state_dict(), model_save_path)
    torch.save({"threshold": final_threshold}, threshold_save_path)

    print(f"\n✅ Training complete.")
    print(f"💾 Model checkpoint saved to: {model_save_path}")
    print(f"💾 Calibrated threshold ({final_threshold:.4f}) saved to: {threshold_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train BDH consistency reasoning network.")
    project_root = find_project_root()
    parser.add_argument("--data-path", type=Path, default=project_root / "data" / "train.csv")
    parser.add_argument("--books-dir", type=Path, default=project_root / "Books")
    parser.add_argument("--output-dir", type=Path, default=project_root)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    train(
        data_path=args.data_path,
        books_dir=args.books_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
