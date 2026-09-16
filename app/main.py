"""FastAPI service for semantic movie search.

Run from the repository root:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000.

Endpoints:
    GET  /         browser UI
    GET  /health   which model / artifacts are loaded
    POST /search   {"text": "...", "top_k": 5} -> top-k movies
    POST /judge    optional LLM-as-a-judge estimate (needs OPENROUTER_API_KEY)
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .env import load_dotenv

load_dotenv()  # must run before retrieval reads its configuration

from . import retrieval  # noqa: E402  (loads the model and index once)

STATIC_DIR = Path(__file__).resolve().parent / "static"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_SYSTEM_PROMPT = """You are an evaluator assistant for a movie recommendation system. Your job is to estimate how relevant and appropriate the recommended movies are for a user's search query.

Use the following guidelines:
- Focus on thematic and plot similarity.
- Do not be overly strict: the recommendations are based on plot embeddings and may not be perfect.
- Give the benefit of the doubt if the results are reasonably close in theme or subject matter.
- Return a percentage estimate from 0% to 100% indicating how well the results match the query.
- Do NOT include any explanation, just the number with a percent sign like: "78%".
- If the movies all clearly relate to the query, even if not exact, that can still be 80%+.
- If only 1 or 2 match decently, give 30-60%.
- If none are even close, go below 30%."""

app = FastAPI(title="Movie semantic search", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Free-text description of a movie")
    top_k: int = Field(5, ge=1, le=50)


class JudgeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    titles: list[str] = Field(..., min_length=1, max_length=50)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "retrieval": retrieval.describe(),
        "judge_enabled": bool(os.getenv("OPENROUTER_API_KEY")),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict:
    return {"results": retrieval.search(request.text, request.top_k)}


@app.post("/judge")
def judge(request: JudgeRequest) -> dict:
    """Ask an LLM how relevant the returned titles look for the query.

    This is a rough sanity signal for the demo UI, not part of the evaluation
    reported in the README (that uses Hit@K / MRR on a labelled test set).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured")

    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'Query: "{request.query}"\nResults: {", ".join(request.titles)}\n'
                "How accurate are these recommendations?",
            },
        ],
    }
    http_request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=30) as response:
            data = json.load(response)
        estimate = data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Judge request failed: {exc}") from exc
    return {"estimate": estimate}
