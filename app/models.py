"""
Pure-PyTorch Transformer language model – built from scratch.

You can:  1) import TransformerLanguageModel in app.main
          2) run app.train to train your own weights
"""
import math
import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────
# Config dataclass with dynamic vocab and special tokens
# ────────────────────────────────────────────────────────────
class TransformerConfig:
    def __init__(self, vocab_size: int,
                 max_seq_len: int = 128,
                 d_model: int = 768,
                 n_heads: int = 12,
                 num_layers: int = 8,
                 d_ff: int = 3072,
                 dropout: float = 0.1):
        # vocabulary size including special tokens
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        # special token IDs (must match TextDataset/tokenizer)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

# ────────────────────────────────────────────────────────────
# Positional encoding
# ────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,T,d_model)
        return x + self.pe[:, : x.size(1)]

# ────────────────────────────────────────────────────────────
# Core Transformer LM
# ────────────────────────────────────────────────────────────
class TransformerLanguageModel(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,      # (B,T,*) ordering
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) longs
        x = self.token_emb(idx)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Stochastic decoding: temperature-scaled, top-k + top-p sampling.
        """
        device = next(self.parameters()).device
        out = prompt_ids.clone().to(device)

        for _ in range(max_new):
            # 1) get logits for last token
            logits = self(out[:, -self.cfg.max_seq_len:])  # (B,T,V)
            logits = logits[:, -1, :] / temperature         # (B, V)

            # 2) top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, top_k, dim=-1)
                min_vals = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_vals, -float("Inf"), logits)

            # 3) nucleus (top-p) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                indices_to_remove = mask.scatter(-1, sorted_indices, mask)
                logits = logits.masked_fill(indices_to_remove, -float("Inf"))

            # 4) sample and append
            probs = torch.softmax(logits, dim=-1)           # (B, V)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            out = torch.cat([out, next_id], dim=1)           # (B, T+1)

        return out