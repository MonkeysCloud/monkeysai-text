"""
Very small training loop: real-text data → prove convergence.
Run with:  poetry run python -m app.train
"""
import os
import time
import torch
from torch.utils.data import DataLoader
from app.models import TransformerConfig, TransformerLanguageModel
from app.dataset import TextDataset

# Import tokenizer for sampling during training
from app.tokenizer import SimpleTokenizer

# ────────────────────────────────────────────────────────────
def main():
    # 1) Configuration using our crawled pages
    import pandas as pd

    # load the scraped texts from Parquet
    df = pd.read_parquet("services/text/data/snapshot.parquet")

    # ─ use the Parquet you already wrote ─
    parquet_path = "services/text/data/snapshot.parquet"
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Expected Parquet at {parquet_path}")
    # feed that directly into TextDataset
    ds = TextDataset(parquet_path, 128)

    tokenizer = ds.tokenizer
    # ─── scale up model to ~140 M parameters ───
    cfg = TransformerConfig(
        # vocab_size=len(tokenizer.token2id),
        vocab_size=16000,
        max_seq_len=256,
        d_model=1024,  # hidden size
        n_heads=16,    # number of attention heads
        num_layers=12, # number of Transformer layers
        d_ff=4096,    # feed-forward layer size
        dropout=0.1,  # dropout rate
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) Model instantiation
    model = TransformerLanguageModel(cfg).to(DEVICE)

    # 3) DataLoader
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    # 4) Optimizer & loss (ignore PAD=0)
    opt     = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # 5) Training loop
    epochs = 50  # increase number of epochs for better learning
    for epoch in range(1, epochs + 1):
        tot = 0.0
        t0 = time.time()
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, cfg.vocab_size), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        avg_loss = tot / len(dl)
        print(f"epoch {epoch}/{epochs}  loss {avg_loss:.3f}  time {time.time()-t0:.1f}s")

        # Save checkpoint each epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/epoch{epoch}.pt")

        # 6) Sample evaluation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            # Prepare prompt with BOS token
            prompt_ids = [cfg.bos_token_id] + tokenizer.encode("MonkeysCMS stores rich-text")
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
            sample = model.generate(
                prompt,
                max_new=16,
                temperature=0.7,
                top_k=100,
                top_p=0.95
            )[0].tolist()
            # Strip BOS and EOS (assuming EOS added by generation or dataset)
            decoded = tokenizer.decode(sample[1:])
            print("Sample:", decoded)

if __name__ == "__main__":
    main()