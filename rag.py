
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SIMILARITY = 0.20  # cosine similarity floor; below this, treat as "no match"

model = SentenceTransformer(MODEL_NAME)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings so inner product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


def create_index(chunks: List[str]):
    """
    Build a cosine-similarity FAISS index over the given chunks.

    Returns:
        (index, embeddings) — embeddings are the normalized vectors
        actually stored in the index.
    """
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = _normalize(embeddings.astype(np.float32))

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(
    query: str,
    chunks: List[str],
    index,
    k: int = 5,
    min_similarity: float = MIN_SIMILARITY,
) -> List[str]:
    """
    Retrieve the top-k most semantically similar chunks to the query.

    Chunks below `min_similarity` cosine similarity are excluded rather
    than always returning k results — this reduces irrelevant context
    being fed to answer generation when the document genuinely doesn't
    contain relevant information.
    """
    results = retrieve_with_scores(query, chunks, index, k, min_similarity)
    return [chunk for chunk, _score in results]


def retrieve_with_scores(
    query: str,
    chunks: List[str],
    index,
    k: int = 5,
    min_similarity: float = MIN_SIMILARITY,
) -> List[Tuple[str, float]]:
    """Same as retrieve(), but also returns the cosine similarity score."""
    if not chunks:
        return []

    q_embed = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_embed = _normalize(q_embed.astype(np.float32))

    k = min(k, len(chunks))
    scores, indices = index.search(q_embed, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < min_similarity:
            continue
        results.append((chunks[idx], float(score)))

    return results


def rerank(query: str, results: List[str]) -> List[str]:
    """
    Rerank a candidate set of chunks by cosine similarity to the query.

    Uses a single batched encode() call for all candidate chunks instead
    of encoding them one at a time, which is both faster and produces
    identical (deterministic) scores.
    """
    if not results:
        return []

    q_embed = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_embed = _normalize(q_embed.astype(np.float32))[0]

    c_embeds = model.encode(results, convert_to_numpy=True, show_progress_bar=False)
    c_embeds = _normalize(c_embeds.astype(np.float32))

    scores = c_embeds @ q_embed  # cosine similarity, since both are normalized
    order = np.argsort(-scores)

    return [results[i] for i in order]
