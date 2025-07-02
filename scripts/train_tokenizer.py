from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import os

# 1) Load your scraped corpus
df = pd.read_parquet("services/text/data/snapshot.parquet")
texts = df["text"].dropna().tolist()

# write out a plain text file for tokenizer training
os.makedirs("services/text/data", exist_ok=True)
txt_path = "services/text/data/nav_corpus.txt"
with open(txt_path, "w", encoding="utf8") as f:
    for t in texts:
        f.write(t.replace("\n", " ") + "\n")

# 2) Train the BPE tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[txt_path],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

# 3) Ensure the output folder exists, then save vocab & merges
out_dir = "services/text/data"
os.makedirs(out_dir, exist_ok=True)
tokenizer.save_model(out_dir, prefix="tokenizer")
print("Tokenizer trained and saved to services/text/data/")