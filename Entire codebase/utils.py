import torch

def tokenize(text):
    return torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

def make_chunks(tokens, chunk_size=512, overlap=128):
    chunks, i = [], 0
    stride = chunk_size - overlap
    while i < len(tokens) - 1:
        chunks.append(tokens[i:i + chunk_size])
        i += stride
    return chunks

def condition_on_character(model, book_state, character, device="cpu"):
    prompt = (
        f"The following statements concern the character {character}. "
        f"Only consider facts and history established in the story."
    )
    tokens = tokenize(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tokens, state=book_state)[2]
