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

# ─────────────────────────────────────────────────────────────────
def main():
    # 1) Prepare dataset using BPE tokenizer inside TextDataset
    parquet_path = "services/text/data/snapshot.parquet"
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Expected Parquet at {parquet_path}")
    seq_len = 256
    ds = TextDataset(parquet_path, seq_len)

    # tokenizer built into dataset (BPE)
    tokenizer = ds.tokenizer

    # 2) Configuration matching training setup
    cfg = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=seq_len,
        d_model=1024,
        n_heads=16,
        num_layers=12,
        d_ff=4096,
        dropout=0.1
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 3) Model instantiation
    model = TransformerLanguageModel(cfg).to(DEVICE)

    # 4) DataLoader with padded BPE IDs
    dataloader = DataLoader(ds, batch_size=32, shuffle=True)

    # 5) Optimizer & loss (ignore PAD=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # 6) Training loop
    epochs = 50
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        start_time = time.time()
        model.train()
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, cfg.vocab_size), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch}/{epochs}  loss {avg_loss:.3f}  time {time.time()-start_time:.1f}s")

        # 7) Save checkpoint each epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/epoch{epoch}.pt")

        # 8) Sample evaluation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            prompt = torch.tensor([ [cfg.bos_token_id] + tokenizer.encode("MonkeysCMS stores rich-text").ids ],
                                  dtype=torch.long, device=DEVICE)
            sample_ids = model.generate(
                prompt,
                max_new=16,
                temperature=0.7,
                top_k=100,
                top_p=0.95
            )[0].tolist()
            # Strip BOS/EOS and decode
            if cfg.eos_token_id in sample_ids:
                sample_ids = sample_ids[: sample_ids.index(cfg.eos_token_id) + 1]
            print("Sample:", tokenizer.decode(sample_ids[1:]))

if __name__ == "__main__":
    main()