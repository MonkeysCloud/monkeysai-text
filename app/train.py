"""
Production‑grade training loop for MonkeysAI‑Text
Run with:  poetry run python -m app.train
Assumptions:
  • Large cleaned corpus (≥10 GB) in services/text/data/snapshot.parquet
  • Byte‑Level BPE tokenizer already trained & saved in services/text/data/
  • CUDA or MPS GPU available; falls back to CPU
"""
import os
import math
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from .models import TransformerConfig, TransformerLanguageModel
from .dataset import TextDataset


# ─────────────────────────────────────────────────────────────────
BATCH_SIZE        = 64          # per gradient step
GRAD_ACCUM_STEPS  = 4           # effective batch = 64 × 4 = 256
SEQ_LEN           = 256
TOTAL_EPOCHS      = 3           # full‑corpus passes; adjust as corpus grows
LR                = 3e-4        # peak LR
WARMUP_STEPS      = 1_000       # linear warm‑up before cosine decay
CHECKPOINT_DIR    = "checkpoints"


def main() -> None:
    parquet_path = "services/text/data/snapshot.parquet"
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Expected Parquet at {parquet_path}")

    # 1) Dataset using BPE tokenizer
    ds = TextDataset(parquet_path, SEQ_LEN)
    tokenizer = ds.tokenizer

    # 2) Config matches inference setup
    cfg = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=SEQ_LEN,
        d_model=1024,
        n_heads=16,
        num_layers=12,
        d_ff=4096,
        dropout=0.1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TransformerLanguageModel(cfg).to(device)

    # 3) DataLoader (no padding needed—TextDataset already pads)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 4) Optimiser, scheduler, AMP scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = TOTAL_EPOCHS * math.ceil(len(ds) / BATCH_SIZE / GRAD_ACCUM_STEPS)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler      = GradScaler(enabled=(device=="cuda"))
    loss_fn     = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # 5) Training loop
    step = 0
    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_loss, t0 = 0.0, time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                logits = model(xb)
                loss   = loss_fn(logits.view(-1, cfg.vocab_size), yb.view(-1)) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            step += 1

        print(f"epoch {epoch}/{TOTAL_EPOCHS}  loss {epoch_loss/len(dl):.3f}  time {time.time()-t0:.1f}s")

        # 6) Checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = f"{CHECKPOINT_DIR}/epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print("Saved", ckpt_path)

        # 7) Quick qualitative sample every epoch
        model.eval()
        prompt_ids = [cfg.bos_token_id] + tokenizer.encode("MonkeysCMS stores rich-text").ids
        with torch.no_grad():
            sample_ids = model.generate(
                torch.tensor([prompt_ids], device=device),
                max_new=32,
                temperature=0.6,
                top_k=40,
                top_p=0.9,
            )[0].tolist()
        if cfg.eos_token_id in sample_ids:
            sample_ids = sample_ids[: sample_ids.index(cfg.eos_token_id) + 1]
        print("Sample:", tokenizer.decode(sample_ids[1:]))


if __name__ == "__main__":
    main()
