"""Semantic Retriever Tool — accepts a user query, generates its embedding,
and performs cosine-similarity search against the FAISS index.

Exposes `semantic_retriever_tool` as a LangChain @tool for the Retrieval Agent.
"""

from __future__ import annotations

import logging

import numpy as np
from langchain_core.tools import tool

from backned.mcp.tools.rag_tools.vector_indexer_tool import embed_query, load_index

logger = logging.getLogger(__name__)


@tool
def semantic_retriever_tool(query: str, top_k: int = 10) -> dict:
    """Perform semantic similarity search for a user query against the invoice index.

    Args:
        query: Natural language question from the user.
        top_k: Number of candidate chunks to return (before re-ranking).

    Returns:
        dict with keys: query (str), chunks (list of dicts with text/source/score),
        total_indexed (int), status (str).
    """
    result = load_index()
    if result is None:
        logger.warning("FAISS index not found — run indexing first")
        return {
            "query":         query,
            "chunks":        [],
            "total_indexed": 0,
            "status":        "index_not_found",
        }

    index, chunks, _ = result
    total = index.ntotal

    if total == 0:
        return {"query": query, "chunks": [], "total_indexed": 0, "status": "empty_index"}

    #  Embed query 
    q_vec = embed_query(query)   # shape (1, 1536), already L2-normalised

    #  FAISS inner-product search (= cosine on normalised vecs) 
    k      = min(top_k, total)
    scores, indices = index.search(q_vec, k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        retrieved.append({
            "text":     chunk["text"],
            "source":   chunk["source"],
            "score":    float(score),
            "chunk_id": int(idx),
        })

    logger.info(
        "Retrieval: query='%.60s' | top-%d returned | top score=%.3f",
        query, len(retrieved), retrieved[0]["score"] if retrieved else 0,
    )

    return {
        "query":         query,
        "chunks":        retrieved,
        "total_indexed": total,
        "status":        "ok",
    }
