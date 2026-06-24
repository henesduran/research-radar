"""Offline tests for the RAG layer (mcp_server/rag.py).

No network and no real ChromaDB on disk: we monkeypatch the embedding call with a
deterministic fake (a tiny bag-of-words vector) and use Chroma's in-memory
EphemeralClient. This exercises our own indexing/dedupe/retrieval logic.
"""

from __future__ import annotations

import pytest

from mcp_server import rag

_VOCAB = ["rag", "graph", "audio"]


@pytest.fixture(autouse=True)
def offline_rag(monkeypatch):
    import chromadb

    # Fresh in-memory vector store per test.
    monkeypatch.setattr(rag, "_chroma_client", chromadb.EphemeralClient())

    # Deterministic fake embeddings: count of each vocab word (+1 padding dim so a
    # doc is never the all-zero vector).
    def fake_embed(texts, task_type):
        return [[float(t.lower().count(w)) for w in _VOCAB] + [1.0] for t in texts]

    monkeypatch.setattr(rag, "_embed", fake_embed)


def _papers():
    return [
        {"id": "a", "title": "RAG evaluation", "summary": "retrieval augmented generation rag rag", "authors": ["X"], "url": "u-a"},
        {"id": "b", "title": "Graph neural nets", "summary": "graph graph networks", "authors": ["Y"], "url": "u-b"},
        {"id": "c", "title": "Audio models", "summary": "audio audio synthesis", "authors": ["Z"], "url": "u-c"},
    ]


def test_index_counts():
    out = rag.index_papers("t", _papers())
    assert out == {"indexed": 3, "total": 3}


def test_index_dedupes():
    rag.index_papers("t", _papers())
    out = rag.index_papers("t", _papers())  # same papers again
    assert out["indexed"] == 0
    assert out["total"] == 3


def test_query_returns_closest_first():
    rag.index_papers("t", _papers())
    res = rag.query("t", "rag rag retrieval", k=2)
    assert res, "expected at least one hit"
    assert res[0]["id"] == "a"  # the RAG paper is closest to a RAG question
    assert set(res[0]) >= {"id", "title", "url", "authors", "text", "distance"}


def test_query_empty_topic_returns_empty():
    assert rag.query("never-indexed-topic", "anything") == []
