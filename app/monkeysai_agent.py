"""monkeysai_agent.py — STREAMING VERSION with JSON extraction and fallback
================================================
Self-contained agent that can:
  1. Semantic search your Qdrant KB (`kb_search`).
  2. Fallback scrape the public web (`web_search`).
  3. Reason step-by-step via a local Llama.cpp model and **stream live events**:
     • `thought`  – before calling a tool
     • `tool`     – after tool returns
     • `token`    – answer tokens
     • `done`     – completion with sources

Drop into `services/text/app/monkeysai_agent.py` and import from `app/main.py` as:
    from .monkeysai_agent import Agent
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Literal, Tuple

import httpx
import trafilatura
from llama_cpp import Llama  # pip install llama-cpp-python
from qdrant_client import QdrantClient  # pip install qdrant-client
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("LLM_MODEL", "models/Mistral-7B-Instruct-Q4_0.gguf"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "monkeysai")
BRAVE_SEARCH_KEY = os.getenv("BRAVE_SEARCH_KEY")
MAX_WEB_CHARS = 4_000

# ────────────────────────────────────────────────────────────────
# Lazy singletons
# ────────────────────────────────────────────────────────────────
_llm: Llama | None = None
_embed: SentenceTransformer | None = None
_qdrant: QdrantClient | None = None
_http: httpx.AsyncClient | None = None

def llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(model_path=str(MODEL_PATH), n_ctx=8192, n_threads=os.cpu_count())
    return _llm


def embed() -> SentenceTransformer:
    global _embed
    if _embed is None:
        _embed = SentenceTransformer(EMBED_MODEL)
    return _embed


def db() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(QDRANT_URL)
    return _qdrant


def http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=30)
    return _http

# ────────────────────────────────────────────────────────────────
# Tools
# ────────────────────────────────────────────────────────────────

def kb_search(query: str, k: int = 5) -> Tuple[str, List[str]]:
    """Semantic search private KB."""
    vec = embed().encode(query, normalize_embeddings=True)
    hits = db().search(
        collection_name=COLLECTION, query_vector=vec, limit=k
    )
    passages, urls = [], []
    for h in hits:
        passages.append(h.payload.get("text", ""))
        if u := h.payload.get("url"):
            urls.append(u)
    return "\n\n".join(passages), urls


DUCK_HTML = "https://html.duckduckgo.com/html/"

async def _duckduckgo(query: str) -> str | None:
    r = await http().post(DUCK_HTML, data={"q": query})
    r.raise_for_status()
    m = re.search(r"result__a\" href=\"(.*?)\"", r.text)
    return m.group(1) if m else None

async def _brave(query: str) -> str | None:
    if not BRAVE_SEARCH_KEY:
        return None
    r = await http().get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": BRAVE_SEARCH_KEY},
        params={"q": query, "source": "news"},
    )
    if r.status_code == 401:
        return None
    results = r.json().get("web", {}).get("results", [])
    return results[0]["url"] if results else None

async def _fetch(url: str) -> Tuple[str, str]:
    r = await http().get(url, follow_redirects=True)
    r.raise_for_status()
    txt = trafilatura.extract(r.text) or ""
    return txt[:MAX_WEB_CHARS], url

async def web_search(query: str) -> Tuple[str, List[str]]:
    """Public web fallback via DuckDuckGo or Brave."""
    url = await _brave(query) or await _duckduckgo(query)
    if not url:
        return "", []
    try:
        text, url = await _fetch(url)
        return text, [url]
    except Exception:
        return "", []

# ────────────────────────────────────────────────────────────────
# Agent with JSON extraction and fallback
# ────────────────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are MonkeysAI.  \n"
    "For every user question, you **must** call `kb_search` with the exact user query first.  \n"
    "If the result is empty, call `web_search` with the same query.  \n"
    "Once you have retrieved context, think step by step but only return the final answer.  \n"
    "Cite any sources you used.  \n"
    "When you want to use a tool, emit JSON:  \n"
    "  {\"tool\":\"kb_search\",\"query\":\"...\"}  \n"
    "or  \n"
    "  {\"tool\":\"web_search\",\"query\":\"...\"}  \n"
    "When you have enough context, emit JSON:  \n"
    "  {\"final\":\"<your answer>\",\"sources\":[\"url1\",\"url2\"]}"
)

ToolName = Literal["kb_search", "web_search"]
Event = Dict[str, object]

class Agent:
    """RAG agent streaming thoughts, tool calls, tokens, and done."""

    def __init__(self):
        self.hist: List[Tuple[str, str]] = []

    async def _llm_json(self, prompt: str) -> dict:
        raw = llm().create_completion(prompt=prompt, stream=False)["choices"][0]["text"]
        txt = raw.strip()
        # Attempt JSON extraction
        if txt.startswith("{"):
            end = txt.rfind("}")
            if end != -1:
                candidate = txt[: end+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        # Fallback: whole output as final
        return {"final": txt, "sources": []}

    async def chat_events(self, user_msg: str) -> AsyncGenerator[Event, None]:
        self.hist.append(("user", user_msg))
        collected: List[str] = []
        while True:
            prompt = (
                SYSTEM_MSG + "\n" + "\n".join(f"{r}: {c}" for r, c in self.hist) + "\nassistant: "
            )
            resp = await self._llm_json(prompt)
            # If tool call requested
            if "tool" in resp:
                tool = resp["tool"]  # type: ignore
                query = resp.get("query", "")
                yield {"type": "thought", "text": f"Calling {tool} with '{query}'"}
                if tool == "kb_search":
                    passages, urls = kb_search(query)
                else:
                    passages, urls = await web_search(query)
                self.hist.append(("assistant", f"TOOL[{tool}]:\n{passages}"))
                collected.extend(urls)
                yield {"type": "tool", "name": tool, "output": passages[:250] + ("…" if len(passages)>250 else "")}
                continue
            # Final answer
            answer = resp.get("final", "")
            sources = resp.get("sources", [])
            async for part in _yield_tokens(answer):
                yield {"type": "token", "text": part}
            yield {"type": "done", "sources": sources}
            break

    async def chat(self, user_msg: str) -> Tuple[str, List[str]]:
        final, sources = "", []
        async for ev in self.chat_events(user_msg):
            if ev["type"] == "token":
                final += ev["text"]
            elif ev["type"] == "done":
                sources = ev.get("sources", [])  # type: ignore
        return final, sources

# ────────────────────────────────────────────────────────────────
# Token streaming helper
# ────────────────────────────────────────────────────────────────
async def _yield_tokens(text: str, delay: float = 0.0) -> AsyncGenerator[str, None]:
    for match in re.findall(r"\S+|\s+", text):
        yield match
        if delay:
            await asyncio.sleep(delay)

# ────────────────────────────────────────────────────────────────
# CLI for quick testing
# ────────────────────────────────────────────────────────────────
async def _cli():
    agent = Agent()
    print("MonkeysAI streaming CLI — Ctrl-C to exit")
    try:
        while True:
            q = input("\n>>> ")
            async for ev in agent.chat_events(q):
                t = ev.get("type")
                if t == "token":
                    print(ev["text"], end="", flush=True)
                elif t == "thought":
                    print(f"\n# {ev['text']}")
                elif t == "tool":
                    print(f"\n[tool {ev['name']} done]")
                elif t == "done":
                    print("\nSources:")
                    for u in ev.get("sources", []):
                        print(" •", u)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(_cli())