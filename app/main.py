import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tokenizers import ByteLevelBPETokenizer
import glob

# 1) Load BPE tokenizer (trained on your corpus)
tokenizer = ByteLevelBPETokenizer(
    "services/text/data/tokenizer-vocab.json",
    "services/text/data/tokenizer-merges.txt",
)

# 2) Load checkpoint (or auto-find latest)
ckpt = os.getenv("CKPT")
if not ckpt:
    # auto-discover latest checkpoint if CKPT is not set
    ckpts = sorted(glob.glob("checkpoints/epoch*.pt"))
    if not ckpts:
        raise RuntimeError("No checkpoint found in checkpoints/ directory. Please train the model or set CKPT env var.")
    ckpt = ckpts[-1]
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dict = torch.load(ckpt, map_location=device)
# checkpoint token embedding shape: [vocab_size, d_model]
ckpt_vocab_size, d_model = state_dict['token_emb.weight'].shape
# checkpoint positional encoding shape: [1, max_seq_len, d_model]
_, max_seq_len, _ = state_dict['pos_enc.pe'].shape

from .models import TransformerConfig, TransformerLanguageModel

# 3) Configuration must exactly match the trained model
cfg = TransformerConfig(
    vocab_size=ckpt_vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    n_heads=16,    # should match training
    num_layers=12, # should match training
    d_ff=4096,
    dropout=0.1,
)

# 4) Instantiate and load weights
model = TransformerLanguageModel(cfg).to(device)
model.load_state_dict(state_dict)
model.eval()

# 5) FastAPI app
app = FastAPI(title="MonkeysAI-Text (BPE)", version="0.0.1")

class TextReq(BaseModel):
    text: str          = Field(..., min_length=1)
    max_tokens: int    = Field(32, ge=1, le=128)
    temperature: float = Field(1.0, gt=0.0)
    top_k: int         = Field(50, ge=0)
    top_p: float       = Field(0.9, gt=0.0, lt=1.0)

class TextResp(BaseModel):
    text: str

@app.post("/generate", response_model=TextResp)
def generate(req: TextReq):
    try:
        # 6) Encode and prepend BOS
        enc = tokenizer.encode(req.text)
        ids = [cfg.bos_token_id] + enc.ids
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        # 7) Generate tokens
        out_ids = model.generate(
            inp,
            max_new=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )[0].tolist()

        # 8) Strip EOS and decode
        if cfg.eos_token_id in out_ids:
            out_ids = out_ids[: out_ids.index(cfg.eos_token_id) + 1]
        decoded = tokenizer.decode(out_ids[1:])
        return TextResp(text=decoded)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))