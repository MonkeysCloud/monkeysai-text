"""
Production-grade training loop for MonkeysAI-Text
Run with:  poetry run python -m app.train  --batch-size 32 --lr 4e-4

Assumptions
-----------
• Large cleaned corpus (≥10 GB) in services/text/data/snapshot.parquet
• Byte-Level BPE tokenizer already trained & saved in services/text/data/
• CUDA or MPS GPU available; falls back to CPU
"""
from __future__ import annotations
import os, math, time, json, argparse, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .models import TransformerConfig, TransformerLanguageModel
from .dataset import TextDataset

# ───────────────────────── defaults ────────────────────────────
DEFAULTS: dict[str, object] = dict(
    batch_size       = 64,
    grad_accum_steps = 4,
    seq_len          = 256,
    max_epochs       = 10,
    lr               = 3e-4,
    warmup_steps     = 1_000,
    checkpoint_dir   = "checkpoints",
    patience         = 5,
    min_delta        = 0.002,   # absolute val-loss drop
    val_split        = 0.02,    # 2 % validation
    workers          = 8,
    seed             = 42,
)

# ───────────────────────── helper funcs ────────────────────────
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ───────────────────────── train core ──────────────────────────
def train(**cfg_args):
    hp = {**DEFAULTS, **cfg_args}
    _set_seeds(hp["seed"])

    parquet_path = "services/text/data/snapshot.parquet"
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Expected Parquet at {parquet_path}")

    # 1) Dataset + val-split
    dataset = TextDataset(parquet_path, hp["seq_len"])
    val_len   = max(1, int(hp["val_split"] * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    tokenizer = train_ds.dataset.tokenizer

    # 2) Model
    cfg = TransformerConfig(
        vocab_size = tokenizer.get_vocab_size(),
        max_seq_len= hp["seq_len"],
        d_model    = 1024,
        n_heads    = 16,
        num_layers = 12,
        d_ff       = 4096,
        dropout    = 0.1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TransformerLanguageModel(cfg).to(device)

    # 3) DataLoaders
    pin   = device == "cuda"
    work  = hp["workers"] if pin else max(2, hp["workers"] // 2)
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                          num_workers=work, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False,
                          num_workers=work, pin_memory=pin)

    # 4) Optimizer + scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                              betas=(0.9, 0.95), weight_decay=0.01)
    tot_steps = hp["max_epochs"] * math.ceil(len(train_ds)
                  / hp["batch_size"] / hp["grad_accum_steps"])
    sched = CosineAnnealingLR(optim, T_max=tot_steps * 2)
    scaler = GradScaler(enabled=device=="cuda")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # — val-pass —
    def eval_epoch() -> float:
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda", enabled=device=="cuda"):
                    logits = model(xb)
                    loss   = loss_fn(logits.view(-1, cfg.vocab_size),
                                     yb.view(-1))
                v_loss += loss.item()
        return v_loss / len(val_dl)

    # 5) Loop
    step, best_val, flat_epochs = 0, None, 0
    for epoch in range(1, hp["max_epochs"] + 1):
        model.train(); optim.zero_grad(set_to_none=True)
        epoch_loss, t0 = 0.0, time.time()

        pb = tqdm(train_dl, desc=f"Epoch {epoch}/{hp['max_epochs']}", leave=False)
        for xb, yb in pb:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(device_type="cuda", enabled=device=="cuda"):
                logits = model(xb)
                loss   = loss_fn(logits.view(-1, cfg.vocab_size),
                                 yb.view(-1)) / hp["grad_accum_steps"]
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * hp["grad_accum_steps"]

            if (step + 1) % hp["grad_accum_steps"] == 0:
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True); sched.step()
            pb.set_postfix(loss=f"{epoch_loss/((step%len(train_dl))+1):.3f}",
                           lr=f"{sched.get_last_lr()[0]:.2e}")
            step += 1
        pb.close()

        # — metrics —
        train_loss = epoch_loss / len(train_dl)
        val_loss   = eval_epoch()
        print(f"epoch {epoch}/{hp['max_epochs']}  "
              f"train {train_loss:.3f}  val {val_loss:.3f}  "
              f"time {(time.time()-t0)/60:.1f} min")

        # — checkpoint —
        Path(hp["checkpoint_dir"]).mkdir(exist_ok=True)
        ckpt = f"{hp['checkpoint_dir']}/epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt); print("Saved", ckpt)

        # — qualitative sample —
        prompt = (
            "Write a 120-word, SEO-optimised introduction paragraph for a web-"
            "agency landing page about MonkeysCloud—our managed DevOps platform. "
            "Include the exact keyword phrase 'managed DevOps platform' once and "
            "'scalable web hosting' once. Use an energetic, authoritative tone and "
            "finish with a clear call to action."
        )
        ids = [cfg.bos_token_id] + tokenizer.encode(prompt).ids
        model.eval()
        with torch.no_grad():
            out = model.generate(torch.tensor([ids], device=device),
                                 max_new=32, temperature=0.6, top_k=40, top_p=0.9)[0].tolist()
        if cfg.eos_token_id in out:
            out = out[:out.index(cfg.eos_token_id)+1]
        print("Sample:", tokenizer.decode(out[1:]))

        # — early-stop —
        if best_val is None or best_val - val_loss > hp["min_delta"]:
            print(f"✅  Val improved: {best_val or val_loss:.3f} → {val_loss:.3f}")
            best_val, flat_epochs = val_loss, 0
        else:
            flat_epochs += 1
            print(f"⏸  No improve ({flat_epochs}/{hp['patience']})")
            if flat_epochs >= hp["patience"]:
                print("🚫  Early-stopping triggered."); break

# ————————————————————————————————————————————————
# CLI shim
# ————————————————————————————————————————————————
if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    p.add_argument("--cfg-json", type=str, help="Path to JSON file of overrides")
    args = vars(p.parse_args())
    if args.pop("cfg_json"):
        args.update(json.loads(Path(args.pop("cfg_json")).read_text()))
    train(**args)