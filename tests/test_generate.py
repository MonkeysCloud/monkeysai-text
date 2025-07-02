from fastapi.testclient import TestClient
from app.main import app, CFG

cli = TestClient(app)

def test_roundtrip():
    payload = {"prompt_ids": [1, 2, 3], "max_tokens": 4}
    r = cli.post("/generate", json=payload)
    assert r.status_code == 200
    out = r.json()["output_ids"]
    assert len(out) == len(payload["prompt_ids"]) + payload["max_tokens"]
    assert max(out) < CFG.vocab_size