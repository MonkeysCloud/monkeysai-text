import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet HF warning in forks

import glob
import json
import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from tokenizers import ByteLevelBPETokenizer

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
CTRL_CLEAN = re.compile(r"[\x00-\x1F]+")  # control chars 0–31

def clean_text(txt: str) -> str:
    return CTRL_CLEAN.sub("", txt)

# -------------------------------------------------------------------
# Load tokenizer & checkpoint
# -------------------------------------------------------------------
TOKENIZER_VOCAB = "services/text/data/tokenizer-vocab.json"
TOKENIZER_MERGE = "services/text/data/tokenizer-merges.txt"

if not (os.path.isfile(TOKENIZER_VOCAB) and os.path.isfile(TOKENIZER_MERGE)):
    raise RuntimeError("Tokenizer files not found; run scripts/train_tokenizer.py first.")

tokenizer = ByteLevelBPETokenizer(TOKENIZER_VOCAB, TOKENIZER_MERGE)

ckpt_path = os.getenv("CKPT") or sorted(glob.glob("checkpoints/epoch*.pt"))[-1]
if not os.path.isfile(ckpt_path):
    raise RuntimeError("Checkpoint not found. Train the model or set CKPT env var.")

device = "cuda" if torch.cuda.is_available() else "cpu"
state_dict = torch.load(ckpt_path, map_location=device)
ckpt_vocab_size, d_model = state_dict["token_emb.weight"].shape
_, max_seq_len, _         = state_dict["pos_enc.pe"].shape

from .models import TransformerConfig, TransformerLanguageModel

cfg = TransformerConfig(
    vocab_size=ckpt_vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    n_heads=16,
    num_layers=12,
    d_ff=4096,
    dropout=0.1,
)

model = TransformerLanguageModel(cfg).to(device)
model.load_state_dict(state_dict)
model.eval()

# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------
app = FastAPI(title="MonkeysAI-Text (BPE)", version="0.1.0")

class TextReq(BaseModel):
    text: str
    max_tokens: int = Field(32, ge=1, le=256)
    temperature: float = Field(0.6, gt=0.0)
    top_k: int = Field(40, ge=0)
    top_p: float = Field(0.9, gt=0.0, lt=1.0)

class TextResp(BaseModel):
    text: str

# -------------------------------------------------------------------
# /generate — blocking response
# -------------------------------------------------------------------
@app.post("/generate", response_model=TextResp)
def generate(req: TextReq):
    try:
        enc = tokenizer.encode(req.text)
        ids = [cfg.bos_token_id] + enc.ids
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        out = model.generate(
            inp,
            max_new=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )[0].tolist()

        if cfg.eos_token_id in out:
            out = out[: out.index(cfg.eos_token_id) + 1]
        return TextResp(text=clean_text(tokenizer.decode(out[1:])))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# -------------------------------------------------------------------
# /generate-stream — SSE streaming
# -------------------------------------------------------------------
@app.post("/generate-stream")
async def generate_stream(req: TextReq):
    def event_source():
        enc = tokenizer.encode(req.text)
        ids = [cfg.bos_token_id] + enc.ids
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        generated = []
        model.eval()
        with torch.no_grad():
            for _ in range(req.max_tokens):
                logits = model(inp)
                next_logits = logits[0, -1] / req.temperature
                probs = torch.softmax(next_logits, dim=-1)
                if req.top_k:
                    vals, idx = torch.topk(probs, req.top_k)
                    probs = torch.zeros_like(probs).scatter_(0, idx, vals)
                next_id = torch.multinomial(probs, 1).item()
                generated.append(next_id)
                yield f"data: {json.dumps({'token': clean_text(tokenizer.decode([next_id]))})}\n\n"
                if next_id == cfg.eos_token_id:
                    break
                inp = torch.cat([inp, torch.tensor([[next_id]], device=device)], 1)
        yield f"data: {json.dumps({'text': clean_text(tokenizer.decode(generated))})}\n\n"
    return StreamingResponse(event_source(), media_type="text/event-stream")
