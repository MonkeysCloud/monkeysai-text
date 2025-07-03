from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import os

# 1) Load your scraped corpus
df = pd.read_parquet("services/text/data/snapshot.parquet")
texts = df["text"].dropna().tolist()

# 2) Write out a raw text file (one doc per line)
os.makedirs("services/text/data", exist_ok=True)
txt_path = "services/text/data/nav_corpus.txt"
with open(txt_path, "w", encoding="utf8") as f:
    for t in texts:
        f.write(t.replace("\n", " ") + "\n")

# 3) Train the BPE tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[txt_path],
    vocab_size=32_000,
    min_frequency=5,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]  # ← fixed quotes
)

# 4) Save vocab & merges
out_dir = "services/text/data"
os.makedirs(out_dir, exist_ok=True)
tokenizer.save_model(out_dir, prefix="tokenizer")

print("Tokenizer trained and saved to services/text/data/")