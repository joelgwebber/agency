from sentence_transformers import SentenceTransformer
from torch import Tensor

_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> Tensor:
    """Simple embedder that uses the HF sentence-transformers model locally."""
    return _embed_model.encode(text)
