"""RAG layer for Research Radar: Gemini embeddings + a ChromaDB vector store.

This is the "memory you can ask questions to." We turn each paper's title+abstract
into an embedding (a list of numbers that captures meaning), store it in a local
ChromaDB collection (one per topic), and answer a question by embedding the question
and asking Chroma for the closest papers.

Why a real vector database (ChromaDB) instead of a Python list?
- Persistence: vectors survive between runs (stored under data/chroma/).
- Speed at scale: Chroma uses approximate nearest-neighbour search.
- Metadata + filtering: we keep each paper's title/url/authors alongside its vector.

We embed documents and queries with DIFFERENT task types
(RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY) - this is a standard RAG practice that
improves retrieval quality with Gemini embeddings.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHROMA_DIR = _ROOT / "data" / "chroma"

# Lazily created so importing this module never needs the network or a running DB.
_genai_client = None
_chroma_client = None


def _genai():
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _genai_client


def _chroma():
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _collection_name(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (topic or "").lower()).strip("-")[:50]
    return f"topic-{slug or 'topic'}"


def _collection(topic: str):
    return _chroma().get_or_create_collection(name=_collection_name(topic))


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Return one embedding vector per input text."""
    from google.genai import types

    resp = _genai().models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in resp.embeddings]


def index_papers(topic: str, papers: list[dict]) -> dict:
    """Embed and store any papers not already in this topic's collection.

    Args:
        topic: groups papers into one collection.
        papers: dicts with at least `id`, `title`, `summary` (and ideally
            `authors`, `url`).

    Returns:
        {"indexed": <how many new>, "total": <collection size>}.
    """
    col = _collection(topic)
    existing = set(col.get()["ids"]) if col.count() else set()
    new = [p for p in (papers or []) if p.get("id") and p["id"] not in existing]
    if not new:
        return {"indexed": 0, "total": col.count()}

    docs = [f"{p.get('title', '')}\n\n{p.get('summary', '')}".strip() for p in new]
    vectors = _embed(docs, "RETRIEVAL_DOCUMENT")
    col.add(
        ids=[p["id"] for p in new],
        embeddings=vectors,
        documents=docs,
        metadatas=[
            {
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "authors": ", ".join((p.get("authors") or [])[:5]),
            }
            for p in new
        ],
    )
    return {"indexed": len(new), "total": col.count()}


def query(topic: str, question: str, k: int = 5) -> list[dict]:
    """Return the k papers whose embeddings are closest to the question.

    Each result: {id, title, url, authors, text, distance}. Smaller distance =
    closer match. Returns [] if the topic has nothing indexed yet.
    """
    col = _collection(topic)
    if not col.count():
        return []
    qvec = _embed([question], "RETRIEVAL_QUERY")[0]
    res = col.query(query_embeddings=[qvec], n_results=min(max(1, k), col.count()))

    out: list[dict] = []
    ids = res["ids"][0]
    for i in range(len(ids)):
        md = (res["metadatas"][0][i] or {}) if res.get("metadatas") else {}
        out.append(
            {
                "id": ids[i],
                "title": md.get("title", ""),
                "url": md.get("url", ""),
                "authors": md.get("authors", ""),
                "text": res["documents"][0][i] if res.get("documents") else "",
                "distance": res["distances"][0][i] if res.get("distances") else None,
            }
        )
    return out
