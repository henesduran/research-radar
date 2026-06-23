"""Research Radar - MCP server.

Exposes the agent's external capabilities as Model Context Protocol (MCP) tools
over stdio. The ADK agents never touch arXiv or the filesystem directly; they go
through these tools. That keeps the boundary explicit and lets us scope which
agent gets which tool (least privilege).

Tools
-----
- search_arxiv:        find recent papers for a topic
- get_seen_paper_ids:  which papers were already covered in past briefs (dedupe)
- record_brief:        persist a finished brief and mark its papers as seen
- list_past_briefs:    metadata about previously generated briefs

Run standalone for a quick smoke test:
    python mcp_server/server.py --selftest "large language model agents"
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
from mcp.server.fastmcp import FastMCP

# Data lives next to the repo root, independent of the process working directory
# (the agent launches this server as a subprocess, so we can't rely on cwd).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_DIR = DATA_DIR / "state"
BRIEFS_DIR = DATA_DIR / "briefs"

# Defensive caps so a malformed agent request can't ask for thousands of results
# or store an unbounded blob.
MAX_RESULTS_CAP = 25
MAX_TOPIC_LEN = 200
MAX_BRIEF_BYTES = 200_000

# Keep the low-level MCP request logs and the arxiv client's page-fetch logs quiet
# so the CLI output stays focused on the brief.
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("arxiv").setLevel(logging.WARNING)

mcp = FastMCP("research-radar")

_arxiv_client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)


def _slug(topic: str) -> str:
    """Filesystem-safe key for a topic."""
    s = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return (s or "topic")[:60]


def _state_path(topic: str) -> Path:
    return STATE_DIR / f"{_slug(topic)}.json"


def _load_state(topic: str) -> dict:
    p = _state_path(topic)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"topic": topic, "seen_ids": [], "briefs": []}


def _save_state(topic: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(topic).write_text(json.dumps(state, indent=2), encoding="utf-8")


@mcp.tool()
def search_arxiv(topic: str, max_results: int = 10, days: int = 0) -> list[dict]:
    """Search arXiv for papers matching a topic, most relevant first.

    Results are ranked by arXiv's relevance score (not date), so on-topic papers
    come first instead of whatever was merely posted most recently.

    Args:
        topic: Natural-language research topic, e.g. "LLM agent planning".
        max_results: How many papers to return (1-25).
        days: If > 0, only include papers submitted within this many days.
            Use 0 (default) for no date filter - best for a literature review.

    Returns:
        A list of papers, most relevant first. Each item has: id (arXiv short id),
        title, authors, published (ISO date), summary, url.
    """
    topic = (topic or "").strip()[:MAX_TOPIC_LEN]
    if not topic:
        return []
    max_results = max(1, min(int(max_results), MAX_RESULTS_CAP))
    days = max(0, min(int(days), 3650))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days) if days else None

    # Over-fetch only when a date filter is active (some relevant hits get
    # dropped); otherwise fetch exactly what we need.
    fetch = min(max_results * 4, 100) if cutoff else max_results
    search = arxiv.Search(
        query=topic,
        max_results=fetch,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[dict] = []
    for r in _arxiv_client.results(search):
        if cutoff and r.published and r.published < cutoff:
            continue  # relevance-sorted, so skip (don't break) out-of-window hits
        papers.append(
            {
                "id": r.get_short_id(),
                "title": r.title.strip().replace("\n", " "),
                "authors": [a.name for a in r.authors][:8],
                "published": r.published.date().isoformat() if r.published else "",
                "summary": r.summary.strip().replace("\n", " ")[:1500],
                "url": r.entry_id,
            }
        )
        if len(papers) >= max_results:
            break
    return papers


@mcp.tool()
def get_seen_paper_ids(topic: str) -> list[str]:
    """Return arXiv ids already covered in past briefs for this topic.

    Use this to skip papers the user has already seen, so each brief only
    surfaces what's new.
    """
    return _load_state(topic).get("seen_ids", [])


@mcp.tool()
def record_brief(topic: str, paper_ids: list[str], brief_markdown: str) -> dict:
    """Persist a finished brief and mark its papers as seen.

    Args:
        topic: The topic this brief covers.
        paper_ids: arXiv ids included in the brief (added to the seen list).
        brief_markdown: The full brief in Markdown.

    Returns:
        {"path": <saved file path>, "total_seen": <int>}.
    """
    brief_markdown = (brief_markdown or "")[:MAX_BRIEF_BYTES]
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"{_slug(topic)}-{stamp}.md"
    (BRIEFS_DIR / fname).write_text(brief_markdown, encoding="utf-8")

    state = _load_state(topic)
    seen = set(state.get("seen_ids", []))
    seen.update(pid for pid in (paper_ids or []) if pid)
    state["seen_ids"] = sorted(seen)
    state["briefs"].append(
        {"date": stamp, "file": fname, "paper_count": len(paper_ids or [])}
    )
    _save_state(topic, state)
    return {"path": str(BRIEFS_DIR / fname), "total_seen": len(state["seen_ids"])}


@mcp.tool()
def list_past_briefs(topic: str) -> list[dict]:
    """List metadata for briefs already generated for this topic (newest last)."""
    return _load_state(topic).get("briefs", [])


def _selftest(topic: str) -> None:
    """Quick offline-ish check that arXiv search works (no MCP transport)."""
    results = search_arxiv(topic, max_results=3, days=120)
    print(f"search_arxiv({topic!r}) -> {len(results)} papers")
    for p in results:
        print(f"  [{p['id']}] {p['title'][:80]}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2] if len(sys.argv) > 2 else "large language model agents")
    else:
        mcp.run()  # stdio transport
