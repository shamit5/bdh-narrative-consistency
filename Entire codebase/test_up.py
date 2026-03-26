import torch
import pandas as pd
from model_up import BDH, BDHConfig

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

device = "cuda" if torch.cuda.is_available() else "cpu"
config = BDHConfig(n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=256)
model = BDH(config).to(device)
model.load_state_dict(torch.load("bdh_trained.pt", map_location=device))
model.eval()
threshold = torch.load("threshold.pt")["threshold"]

df = pd.read_csv("data/test.csv")
df = df[df["content"].notna()].reset_index(drop=True)

BOOK_PATHS = {
    "The Count of Monte Cristo": "Books/The Count of Monte Cristo.txt",
    "In Search of the Castaways": "Books/In search of the castaways.txt",
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
    prompt = f"The following statements concern the character {character}. Only consider facts established in the story."
    tokens = tokenize(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tokens, state=book_state)[2]

results = []
with torch.no_grad():
    for row in df.itertuples():
        book_state = book_states[normalize_book_name(row.book_name)]
        char_state = condition_on_character(model, book_state, row.char)
        tokens = tokenize(row.content).to(device)
        chunks = make_chunks(tokens)
        if not chunks:
            results.append([row.id, 0, "empty"])
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
out_df.to_csv("results.csv", index=False)
print(" Saved results.csv")
