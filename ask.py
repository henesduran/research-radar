"""Research Radar - ask questions about a topic's papers (RAG).

On first use for a topic it indexes that topic's arXiv papers into a local vector
store (embeds their abstracts). Then it answers your question using ONLY those
papers, with citations. Re-asking the same topic reuses the index.

Examples:
    python ask.py "retrieval augmented generation evaluation" "What metrics evaluate RAG?"
    python ask.py "graph neural networks for drug discovery" "What problems do GNNs solve here?"
    python ask.py "diffusion models for audio" "List the main approaches" --reindex
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import warnings

# Robust, quiet output (same reasoning as radar.py).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from mcp_server import rag, server  # noqa: E402
from research_radar.agent import MODEL, researcher  # noqa: E402
from research_radar.errors import explain_error, is_transient  # noqa: E402

APP = "research_radar_ask"


def _ensure_indexed(topic: str, max_results: int, reindex: bool) -> int:
    """Make sure the topic's papers are in the vector store. Returns paper count."""
    have = rag._collection(topic).count()
    if have and not reindex:
        return have
    print(f"Indexing papers for '{topic}' (embedding abstracts, one-time)...")
    papers = server.search_arxiv(topic, max_results=max_results)
    result = rag.index_papers(topic, papers)
    print(f"  indexed {result['indexed']} new, {result['total']} total.")
    return result["total"]


async def _answer(topic: str, question: str) -> None:
    sessions = InMemorySessionService()
    runner = Runner(agent=researcher, app_name=APP, session_service=sessions)
    await sessions.create_session(app_name=APP, user_id="cli", session_id="cli")
    msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Topic: {topic}\nQuestion: {question}")],
    )
    async for event in runner.run_async(
        user_id="cli", session_id="cli", new_message=msg
    ):
        if event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts).strip()
            if text:
                print(f"\n{text}")


async def run(topic, question, max_results, reindex, retries) -> int:
    print(f"\n\U0001f50d Research Radar - Ask  ·  model: {MODEL}")
    print(f"  topic: {topic}\n  question: {question}")
    print("=" * 64)

    try:
        count = _ensure_indexed(topic, max_results, reindex)
    except Exception as err:  # noqa: BLE001
        print("\n" + "=" * 64 + "\n❌ " + explain_error(err))
        return 1
    if not count:
        print("\nNo papers found to index for this topic - nothing to answer from.")
        return 1

    attempt = 0
    while True:
        try:
            await _answer(topic, question)
            print("\n" + "=" * 64)
            return 0
        except Exception as err:  # noqa: BLE001
            if is_transient(err) and attempt < retries:
                attempt += 1
                wait = 5 * attempt
                print(f"\n⏳ Temporary error; retrying in {wait}s "
                      f"(attempt {attempt}/{retries})...")
                await asyncio.sleep(wait)
                continue
            print("\n" + "=" * 64 + "\n❌ " + explain_error(err))
            return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about a topic's arXiv papers (RAG, cited answers).",
    )
    parser.add_argument("topic", help="the research topic (quote it)")
    parser.add_argument("question", nargs="+", help="your question (quote it)")
    parser.add_argument("--max-results", type=int, default=25,
                        help="how many papers to index on first run (default: 25)")
    parser.add_argument("--reindex", action="store_true",
                        help="re-fetch and re-embed the topic's papers")
    parser.add_argument("--retries", type=int, default=2,
                        help="automatic retries on transient (503) errors")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(
        args.topic, " ".join(args.question),
        args.max_results, args.reindex, args.retries,
    )))


if __name__ == "__main__":
    main()
