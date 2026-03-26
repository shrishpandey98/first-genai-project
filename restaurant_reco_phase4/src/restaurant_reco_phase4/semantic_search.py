import logging
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# We'll use a small, fast model suitable for CPU inference
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def compute_semantic_scores(query: str, documents: List[str]) -> List[float]:
    """
    Computes the cosine similarity between the query and a list of documents.
    Returns a list of float scores in [0, 1].
    """
    if not query or not documents:
        return [0.0] * len(documents)

    try:
        model = get_model()
        
        # Compute embeddings
        query_emb = model.encode([query])
        doc_embs = model.encode(documents)
        
        # Compute cosine similarity
        sims = cosine_similarity(query_emb, doc_embs)[0]
        
        # Normalize from [-1, 1] to [0, 1] for our scoring formula
        normalized_sims = [(s + 1.0) / 2.0 for s in sims]
        return normalized_sims
        
    except Exception as e:
        logger.error(f"Failed to compute semantic scores: {e}")
        return [0.0] * len(documents)
