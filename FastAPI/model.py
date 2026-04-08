import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load embeddings and metadata once at module load.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
EMBEDDINGS_PATH = PROJECT_DIR / "notebooks" / "movie_embeddings_2.npy"
METADATA_PATH = PROJECT_DIR / "notebooks" / "movie_metadata_2.pkl"

embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# Normalize embeddings so inner product search behaves like cosine similarity.
faiss.normalize_L2(embeddings)

# Build FAISS index once
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# Load embedding model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_query(text):
    # Embed input text query and normalize it for cosine similarity search.
    embedding = model.encode([text]).astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


def search_similar_movies(query_text, top_k=5):
    query_vec = embed_query(query_text)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        movie = metadata[idx]
        results.append({
            "title": movie["title"],
            "similarity": float(score),
            # add more metadata fields if you want
        })
    return results
