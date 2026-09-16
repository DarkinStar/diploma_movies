"""Dense retrieval over the movie corpus: embedding model + FAISS index.

The model, the pre-computed corpus embeddings and the FAISS index are loaded
once when this module is imported. Which model and which artifacts are used is
controlled by environment variables (see .env.example), so the same code can
serve either the base `mxbai-embed-large-v1` model or a LoRA-fine-tuned one.

Corpus embeddings are produced by notebooks/03_build_index.ipynb. Query and
corpus MUST be encoded by the same model, so the embeddings file has to match
the model configured here (the dimension check below catches the obvious case).
"""
import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]

# mxbai is an asymmetric retrieval model: queries get this prefix, documents do not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODEL_NAME = os.getenv("MOVIE_SEARCH_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
LORA_ADAPTER = os.getenv("MOVIE_SEARCH_LORA_ADAPTER", "")


def _project_path(env_var: str, default: str) -> Path:
    path = Path(os.getenv(env_var, default))
    return path if path.is_absolute() else PROJECT_DIR / path


EMBEDDINGS_PATH = _project_path("MOVIE_SEARCH_EMBEDDINGS", "artifacts/embeddings_mxbai_base.npy")
METADATA_PATH = _project_path("MOVIE_SEARCH_METADATA", "artifacts/metadata.pkl")


def _load_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    if LORA_ADAPTER:
        from peft import PeftModel  # optional dependency, only needed for fine-tuned models

        adapter_dir = PROJECT_DIR / LORA_ADAPTER
        transformer = model._first_module()
        transformer.auto_model = PeftModel.from_pretrained(
            transformer.auto_model, str(adapter_dir)
        ).merge_and_unload()
        log.info("Loaded LoRA adapter from %s", adapter_dir)
    return model


def _load_artifacts() -> tuple[np.ndarray, list[dict]]:
    missing = [p for p in (EMBEDDINGS_PATH, METADATA_PATH) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing retrieval artifacts: "
            + ", ".join(str(p) for p in missing)
            + ". Generate them with notebooks/03_build_index.ipynb (see README)."
        )
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"{METADATA_PATH.name} has {len(metadata)} rows but "
            f"{EMBEDDINGS_PATH.name} has {embeddings.shape[0]} vectors."
        )
    return embeddings, metadata


model = _load_model()
embeddings, metadata = _load_artifacts()

model_dim = model.get_sentence_embedding_dimension()
if model_dim != embeddings.shape[1]:
    raise ValueError(
        f"Model '{MODEL_NAME}' produces {model_dim}-d vectors but "
        f"{EMBEDDINGS_PATH.name} contains {embeddings.shape[1]}-d vectors. "
        "Re-encode the corpus with notebooks/03_build_index.ipynb using the same model."
    )

# Inner product over L2-normalised vectors == cosine similarity.
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
log.info("Index ready: %d movies, dim=%d", index.ntotal, embeddings.shape[1])


def describe() -> dict:
    return {
        "model": MODEL_NAME,
        "lora_adapter": LORA_ADAPTER or None,
        "embeddings": EMBEDDINGS_PATH.name,
        "corpus_size": int(index.ntotal),
        "dim": int(embeddings.shape[1]),
    }


def embed_query(text: str) -> np.ndarray:
    vec = model.encode([QUERY_PREFIX + text]).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def search(query_text: str, top_k: int = 5) -> list[dict]:
    scores, indices = index.search(embed_query(query_text), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        movie = metadata[idx]
        summary = str(movie.get("summary") or "")
        results.append(
            {
                "title": movie.get("title"),
                "release_year": movie.get("release_year"),
                "genre": movie.get("genre"),
                "summary": summary[:300] + ("..." if len(summary) > 300 else ""),
                "similarity": round(float(score), 4),
            }
        )
    return results
