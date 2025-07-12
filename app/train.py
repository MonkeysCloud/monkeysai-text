"""
Production-grade training loop for MonkeysAI-Text
Run with:
    poetry run python -m app.train --batch-size 32 --lr 4e-4
"""
from __future__ import annotations
import os, math, time, json, argparse, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .models import TransformerConfig, TransformerLanguageModel
from .dataset import TextDataset

# ───────────────────────────────────────────────────────────────
DEFAULTS: dict[str, object] = dict(
    batch_size       = 64,
    grad_accum_steps = 4,
    seq_len          = 384,        # longer context
    max_epochs       = 14,         # room for one LR restart
    lr               = 3e-4,
    checkpoint_dir   = "checkpoints",
    patience         = 8,
    min_delta        = 0.001,      # tiny absolute drop
    val_split        = 0.02,
    workers          = 8,
    seed             = 42,
)

# ───────────────────────── helpers ─────────────────────────────
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ───────────────────────── train() ─────────────────────────────
def train(**cfg_args):
    hp = {**DEFAULTS, **cfg_args}
    _set_seeds(hp["seed"])

    parquet = "services/text/data/snapshot.parquet"
    if not os.path.isfile(parquet):
        raise FileNotFoundError(f"Missing {parquet}")

    # 1) Dataset
    full = TextDataset(parquet, hp["seq_len"])
    val_len = max(1, int(hp["val_split"] * len(full)))
    train_ds, val_ds = random_split(full, [len(full)-val_len, val_len])
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
    pin = device == "cuda"
    workers = hp["workers"] if pin else max(2, hp["workers"] // 2)
    train_dl = DataLoader(train_ds, hp["batch_size"], True,  num_workers=workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   hp["batch_size"], False, num_workers=workers, pin_memory=pin)

    # 4) Optimiser + One-Cycle schedule
    optim = torch.optim.AdamW(model.parameters(), lr=hp["lr"], betas=(0.9,0.95), weight_decay=0.01)
    total_steps = hp["max_epochs"] * math.ceil(len(train_ds)/hp["batch_size"]/hp["grad_accum_steps"])
    sched = OneCycleLR(
        optim, max_lr=hp["lr"], total_steps=total_steps,
        pct_start=0.15,  anneal_strategy="cos",
        div_factor=10,   final_div_factor=1e4
    )
    scaler  = GradScaler(enabled=device=="cuda")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # 5) Validation helper
    def val_loss() -> float:
        model.eval(); loss=0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb,yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda", enabled=device=="cuda"):
                    loss += loss_fn(model(xb).view(-1,cfg.vocab_size), yb.view(-1)).item()
        return loss / len(val_dl)

    # 6) Train loop
    step, best, flat, restart_used = 0, None, 0, False
    for epoch in range(1, hp["max_epochs"]+1):
        model.train(); optim.zero_grad(set_to_none=True)
        running, t0 = 0.0, time.time()

        bar = tqdm(train_dl, desc=f"Epoch {epoch}/{hp['max_epochs']}", leave=False)
        for xb, yb in bar:
            xb,yb = xb.to(device), yb.to(device)
            with autocast(device_type="cuda", enabled=device=="cuda"):
                loss = loss_fn(model(xb).view(-1,cfg.vocab_size), yb.view(-1)) / hp["grad_accum_steps"]
            scaler.scale(loss).backward()
            running += loss.item()*hp["grad_accum_steps"]

            if (step+1) % hp["grad_accum_steps"] == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True); sched.step()
            bar.set_postfix(loss=f"{running/((step%len(train_dl))+1):.3f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            step+=1
        bar.close()

        v = val_loss()
        print(f"epoch {epoch}  train {running/len(train_dl):.3f}  val {v:.3f}  "
              f"time {(time.time()-t0)/60:.1f} min")

        Path(hp["checkpoint_dir"]).mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"{hp['checkpoint_dir']}/epoch{epoch}.pt")

        # sample
        prompt = ("Write a 120-word, SEO-optimised introduction paragraph for a web-"
                  "agency landing page about MonkeysCloud—our managed DevOps platform. "
                  "Include 'managed DevOps platform' once and 'scalable web hosting' once.")
        ids = [cfg.bos_token_id]+tokenizer.encode(prompt).ids
        model.eval()
        with torch.no_grad():
            out = model.generate(torch.tensor([ids],device=device), max_new=32,
                                 temperature=0.6, top_k=40, top_p=0.9)[0].tolist()
        if cfg.eos_token_id in out: out = out[:out.index(cfg.eos_token_id)+1]
        print("Sample:", tokenizer.decode(out[1:]))

        # early-stop or warm-restart
        if best is None or best - v > hp["min_delta"]:
            best, flat = v, 0
            print(f"✅  Val improved → {v:.3f}")
        else:
            flat += 1
            print(f"⏸  No improve ({flat}/{hp['patience']})")
            if flat >= hp["patience"]:
                if not restart_used:
                    print("↻  LR warm-restart")
                    sched._step_count = 0; flat = 0; restart_used = True
                else:
                    print("🚫  Early-stopping"); break

# ── CLI shim ───────────────────────────────────────────────────
if __name__ == "__main__":
    arg = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in DEFAULTS.items(): arg.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    arg.add_argument("--cfg-json", type=str, help="Path to JSON overrides")
    ns = vars(arg.parse_args())
    if ns.get("cfg_json"): ns.update(json.loads(Path(ns.pop("cfg_json")).read_text()))
    train(**ns)