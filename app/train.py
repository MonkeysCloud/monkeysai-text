"""
Production-grade training loop for MonkeysAI-Text
Run with:  poetry run python -m app.train
Assumptions:
  • Large cleaned corpus (≥10 GB) in services/text/data/snapshot.parquet
  • Byte-Level BPE tokenizer already trained & saved in services/text/data/
  • CUDA or MPS GPU available; falls back to CPU
"""
import os
import math
import time
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .models import TransformerConfig, TransformerLanguageModel
from .dataset import TextDataset


# ─────────────────────────────────────────────────────────────────
BATCH_SIZE         = 64          # per gradient step
GRAD_ACCUM_STEPS   = 4           # effective batch = 64 × 4 = 256
SEQ_LEN            = 256
TOTAL_EPOCHS       = 10          # upper bound; early-stop usually cuts sooner
LR                 = 3e-4        # peak learning-rate
WARMUP_STEPS       = 1_000       # linear warm-up before cosine decay
CHECKPOINT_DIR     = "checkpoints"

# Early-stopping
EARLY_STOP_PATIENCE = 3          # epochs to wait
MIN_DELTA           = 0.02      # 0.5 % relative improvement


def main() -> None:
    parquet_path = "services/text/data/snapshot.parquet"
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Expected Parquet at {parquet_path}")

    # 1) Dataset + 1 % validation split
    full_ds = TextDataset(parquet_path, SEQ_LEN)
    val_len = max(1, int(0.01 * len(full_ds)))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    tokenizer = train_ds.dataset.tokenizer  # same tokenizer underneath

    # 2) Model config
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

    # 3) DataLoaders
    pin      = device == "cuda"
    workers  = 8 if pin else 4
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=workers, pin_memory=pin)

    # 4) Optimiser, scheduler, AMP scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = TOTAL_EPOCHS * math.ceil(len(train_ds)
                                           / BATCH_SIZE / GRAD_ACCUM_STEPS)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps * 2)
    scaler      = GradScaler(enabled=(device == "cuda"))
    loss_fn     = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # ── helper: validation pass ─────────────────────────────────
    def eval_epoch() -> float:
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits = model(xb)
                    loss   = loss_fn(logits.view(-1, cfg.vocab_size),
                                     yb.view(-1))
                v_loss += loss.item()
        return v_loss / len(val_dl)

    # 5) Training loop with early-stopping
    step           = 0
    best_val       = None
    epochs_no_imp  = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            with autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss   = loss_fn(logits.view(-1, cfg.vocab_size),
                                 yb.view(-1)) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            pbar.set_postfix(
                loss=f"{epoch_loss / ((step % len(train_dl)) + 1):.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )
            step += 1
        pbar.close()

        # ── end-of-epoch bookkeeping ───────────────────────────
        train_loss = epoch_loss / len(train_dl)
        val_loss   = eval_epoch()
        elapsed    = (time.time() - t0) / 60
        print(f"epoch {epoch}/{TOTAL_EPOCHS}  "
              f"train {train_loss:.3f}  val {val_loss:.3f}  "
              f"time {elapsed:.1f} min")

        # 6) Checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = f"{CHECKPOINT_DIR}/epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print("Saved", ckpt_path)

        # 7) Qualitative sample every epoch
        # Use an SEO-optimised seed so samples resemble the copy you'll generate
        # in production (landing pages, product blurbs, etc.).
        prompt_text = (
            "Write a 120-word, SEO-optimised introduction paragraph for a web-"
            "agency landing page about MonkeysCloud—our managed DevOps platform. "
            "Include the exact keyword phrase 'managed DevOps platform' once and "
            "'scalable web hosting' once. Use an energetic, authoritative tone and "
            "finish with a clear call to action."
        )
        prompt_ids = [cfg.bos_token_id] + tokenizer.encode(prompt_text).ids
        with torch.no_grad():
            sample_ids = model.generate(
                torch.tensor([prompt_ids], device=device),
                max_new=32, temperature=0.6, top_k=40, top_p=0.9
            )[0].tolist()
        if cfg.eos_token_id in sample_ids:
            sample_ids = sample_ids[: sample_ids.index(cfg.eos_token_id) + 1]
        print("Sample:", tokenizer.decode(sample_ids[1:]))

        # 8) Early-stopping check (absolute Δ)
        if best_val is None or (best_val - val_loss) > MIN_DELTA:
            best_val = val_loss
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= EARLY_STOP_PATIENCE:
                print("Early-stopping triggered.")
                break


if __name__ == "__main__":
    main()