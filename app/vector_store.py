import json
import logging
import math
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    LOCAL_INDEX_FILE,
)

logger = logging.getLogger(__name__)

# Initialize embedding model lazily or as singleton
_embedder_instance: Optional[HuggingFaceEmbeddings] = None
_pinecone_client = None
_pinecone_index = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Returns the singleton HuggingFaceEmbeddings model."""
    global _embedder_instance
    if _embedder_instance is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedder_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedder_instance


def get_pinecone_index():
    """Initializes and returns the Pinecone index if configured, or None."""
    global _pinecone_client, _pinecone_index

    if _pinecone_index is not None:
        return _pinecone_index

    if not PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY is not set. Pinecone index will not be used directly.")
        return None

    try:
        from pinecone import Pinecone, ServerlessSpec

        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        indexes = _pinecone_client.list_indexes()
        
        # Safe extraction of existing index names
        if hasattr(indexes, "names"):
            existing_names = indexes.names()
        else:
            existing_names = [idx.name if hasattr(idx, "name") else idx.get("name") for idx in indexes]

        if PINECONE_INDEX not in existing_names:
            logger.info(f"Creating Pinecone index: '{PINECONE_INDEX}' with dimension {EMBEDDING_DIMENSION}")
            _pinecone_client.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            logger.info(f"Created Pinecone index '{PINECONE_INDEX}' successfully.")

        _pinecone_index = _pinecone_client.Index(PINECONE_INDEX)
        return _pinecone_index

    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        return None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def query_local_cache(query_vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
    """Queries local vector cache as fallback when Pinecone is not reachable."""
    if not LOCAL_INDEX_FILE.exists():
        logger.warning(f"Local vector cache not found at {LOCAL_INDEX_FILE}. Run ingestion first.")
        return []

    try:
        with open(LOCAL_INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        scored_records = []
        for item in data:
            sim = cosine_similarity(query_vector, item["values"])
            scored_records.append({
                "text": item["metadata"].get("text", ""),
                "page": item["metadata"].get("page", 0),
                "score": round(sim, 4),
            })

        # Sort descending by similarity score
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]
    except Exception as e:
        logger.error(f"Error querying local cache: {e}")
        return []


def query_vector_store(query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Embeds query text and retrieves top_k most similar chunks.
    Prioritizes Pinecone; gracefully falls back to local cache if Pinecone is unconfigured.
    Returns:
        List of dicts: [{"text": str, "page": int, "score": float}, ...]
    """
    embedder = get_embedder()
    query_vector = embedder.embed_query(query_text)

    index = get_pinecone_index()
    if index is not None:
        try:
            res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            results = []
            for match in res.get("matches", []):
                meta = match.get("metadata", {})
                score = match.get("score", 0.0)
                results.append({
                    "text": meta.get("text", ""),
                    "page": meta.get("page", 0),
                    "score": round(float(score), 4),
                })
            if results:
                return results
        except Exception as e:
            logger.warning(f"Pinecone query encountered error ({e}). Attempting local cache fallback.")

    # Fallback to local cache
    return query_local_cache(query_vector, top_k=top_k)
