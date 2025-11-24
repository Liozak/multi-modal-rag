"""
Simple in-memory vector index for text chunks.

STEP 5:
- Build an index from a list of DocumentChunk objects.
- Search the index using cosine similarity.
"""

from typing import List, Tuple, Dict, Any

import numpy as np

from app.models import DocumentChunk
from app.embeddings import embed_texts


def build_text_index(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """
    Given a list of DocumentChunk objects (text modality),
    compute embeddings and return an index object:

    {
      "embeddings": np.ndarray of shape (N, D),
      "chunks": List[DocumentChunk]
    }
    """
    if not chunks:
        raise ValueError("No chunks provided to build_text_index.")

    texts = [c.content for c in chunks]
    embeddings_list = embed_texts(texts)  # List[List[float]]

    embedding_matrix = np.array(embeddings_list, dtype="float32")  # shape: (N, D)

    index = {
        "embeddings": embedding_matrix,
        "chunks": chunks,
    }
    return index


def search_text_index(
    index: Dict[str, Any],
    query: str,
    top_k: int = 5,
) -> List[Tuple[float, DocumentChunk]]:
    """
    Search the index with a query string.
    Returns a list of (similarity_score, DocumentChunk) sorted by score (desc).

    Uses cosine similarity between query embedding and chunk embeddings.
    """
    if "embeddings" not in index or "chunks" not in index:
        raise ValueError("Index is missing 'embeddings' or 'chunks' keys.")

    embeddings = index["embeddings"]  # shape: (N, D)
    chunks = index["chunks"]

    if embeddings.shape[0] == 0:
        return []

    # Embed the query
    query_embedding = np.array(embed_texts([query])[0], dtype="float32")  # shape: (D,)

    # Compute cosine similarity
    # cos_sim = (A · B) / (||A|| * ||B||)
    dot_products = embeddings @ query_embedding  # shape: (N,)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    cosine_similarities = dot_products / norms  # shape: (N,)

    # Get indices of top_k highest similarities
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(-cosine_similarities)[:top_k]

    results: List[Tuple[float, DocumentChunk]] = []
    for idx in top_indices:
        score = float(cosine_similarities[idx])
        chunk = chunks[int(idx)]
        results.append((score, chunk))

    return results
