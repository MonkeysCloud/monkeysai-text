from .dataset import TextDataset
from .tokenizer import SimpleTokenizer
import torch, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1) Load or build tokenizer & dataset
# TextDataset builds SimpleTokenizer internally
# point at the Parquet feed you already wrote:
parquet_path = "services/text/data/snapshot.parquet"
if not os.path.isfile(parquet_path):
    raise FileNotFoundError(f"Couldn’t find {parquet_path}")

ds = TextDataset(parquet_path, 128)
tokenizer = ds.tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

from .models import TransformerConfig, TransformerLanguageModel
# 2) Configuration with dynamic vocab size and matching hyperparameters
cfg = TransformerConfig(
    vocab_size=len(tokenizer.token2id),
    max_seq_len=128,
    # bump up model size:
    d_model=1024,
    n_heads=16,
    num_layers=12,
    d_ff=4096,
    dropout=0.1
)

# 3) Model instantiation and checkpoint loading
model = TransformerLanguageModel(cfg).to(device)
ckpt = os.getenv("CKPT", "")
if ckpt:
    model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

# 4) FastAPI app
app = FastAPI(title="MonkeysAI-Text (scratch)", version="0.0.1")

class TextReq(BaseModel):
    text: str          = Field(..., min_length=1)
    max_tokens: int    = Field(32, ge=1, le=128)
    temperature: float = Field(1.0, gt=0.0, description="Sampling temperature")
    top_k: int         = Field(50, ge=0, description="Top-k sampling (0=disabled)")
    top_p: float       = Field(0.9, gt=0.0, lt=1.0, description="Nucleus sampling threshold")

class TextResp(BaseModel):
    text: str

@app.post("/generate", response_model=TextResp)
def generate(req: TextReq):
    try:
        # 5) Encode input text to IDs, prepend BOS
        ids = [cfg.bos_token_id] + tokenizer.encode(req.text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        # 6) Generate with sampling
        out = model.generate(
            inp,
            max_new=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )[0].tolist()
        # 7) Strip BOS/EOS and decode
        # EOS handling: stop at eos_token_id if present
        if cfg.eos_token_id in out:
            out = out[: out.index(cfg.eos_token_id) + 1]
        decoded_ids = next(tok for tok in [out[1:]] if tok)
        return TextResp(text=tokenizer.decode(decoded_ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))