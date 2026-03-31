import os
from pathlib import Path
import torch
import pandas as pd
import torch.nn.functional as F
from model_up import BDH, BDHConfig

# Root path resolution: works whether run from root or "Entire codebase"
ROOT_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.parent / "data").exists() else Path(".")

def tokenize(text):
    return torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

def make_chunks(tokens, chunk_size=512, overlap=128):
    chunks, i = [], 0
    stride = chunk_size - overlap
    while i < len(tokens) - 1:
        chunks.append(tokens[i:i + chunk_size])
        i += stride
    return chunks

def normalize_book_name(name):
    return name.strip().lower()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = BDHConfig(
    n_layer=4,
    n_embd=128,
    n_head=4,
    mlp_internal_dim_multiplier=8,
    vocab_size=256
)

model = BDH(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Load Train CSV
train_csv_path = ROOT_DIR / "data" / "train.csv"
df = pd.read_csv(train_csv_path)
df["label"] = (
    df["label"].str.strip().str.lower()
    .map({"consistent": 1, "contradict": 0})
)
assert df["label"].isna().sum() == 0
df = df[df["content"].notna()].reset_index(drop=True)
train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)

BOOK_PATHS = {
    "The Count of Monte Cristo": ROOT_DIR / "Books" / "The Count of Monte Cristo.txt",
    "In Search of the Castaways": ROOT_DIR / "Books" / "In search of the castaways.txt",
}

def build_book_state(model, text):
    tokens = tokenize(text).to(device)
    chunks = make_chunks(tokens)
    state = None
    with torch.no_grad():
        for ch in chunks:
            if len(ch) < 2:
                continue
            state = model(ch[:-1].unsqueeze(0), state=state)[2]
    return state.detach()

book_states = {}
for name, path in BOOK_PATHS.items():
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        book_states[normalize_book_name(name)] = build_book_state(model, f.read())

def condition_on_character(model, book_state, character):
    prompt = (
        f"The following statements concern the character {character}. "
        f"Only consider facts and history established in the story."
    )
    tokens = tokenize(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tokens, state=book_state)[2]

epochs = 30
model.train()
epoch_thresholds = []

for ep in range(epochs):
    pos_scores, neg_scores = [], []
    total_loss = 0.0
    for row in train_df.itertuples():
        book_state = book_states[normalize_book_name(row.book_name)]
        char_state = condition_on_character(model, book_state, row.char)
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
    print(f"Epoch {ep+1:02d} | Loss {total_loss:.3f} | Cons {mean_pos:.3f} | Contr {mean_neg:.3f}")

final_threshold = sum(epoch_thresholds[-5:]) / 5 if len(epoch_thresholds) >= 5 else epoch_thresholds[-1]
torch.save(model.state_dict(), ROOT_DIR / "bdh_trained.pt")
torch.save({"threshold": final_threshold}, ROOT_DIR / "threshold.pt")
print("✅ Training complete")
print("✅ Threshold:", final_threshold)
