"""
Functions to create embeddings (vector representations) for text.
We use sentence-transformers here.

This file exposes:
- get_text_embedding_model()
- embed_texts(texts)
"""

from typing import List

from sentence_transformers import SentenceTransformer

# We'll load the model lazily (only when needed)
_text_model = None


def get_text_embedding_model() -> SentenceTransformer:
    """
    Returns a singleton SentenceTransformer model.
    Downloads the model the first time you call it.
    """
    global _text_model
    if _text_model is None:
        # Small, fast model (enough for assignment)
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Takes a list of strings and returns a list of embeddings (vectors).
    Each embedding is a list of floats.
    """
    model = get_text_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    # Convert numpy array → Python lists for simplicity
    return embeddings.tolist()
