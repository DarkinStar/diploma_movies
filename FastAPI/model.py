import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load embeddings and metadata once at module load
embeddings = np.load("movie_embeddings.npy")
with open("movie_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Build FAISS index once
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Load embedding model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(text):
    # Embed input text query and reshape for FAISS search
    embedding = model.encode([text])
    return embedding

def search_similar_movies(query_text, top_k=5):
    query_vec = embed_query(query_text)
    D, I = index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        movie = metadata[idx]
        results.append({
            "title": movie['title'],
            "distance": float(dist),
            # add more metadata fields if you want
        })
    return results