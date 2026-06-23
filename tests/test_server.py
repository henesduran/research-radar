"""Unit tests for the MCP server's tools and helpers.

These run fully offline - no network, no LLM. The one tool that normally hits the
network (`search_arxiv`) is tested with a fake arXiv client so we exercise our own
logic (clamping, shaping, filtering) deterministically.

Run:  pytest -q
"""

from __future__ import annotations

import datetime as dt

import pytest

from mcp_server import server


@pytest.fixture(autouse=True)
def temp_data_dirs(tmp_path, monkeypatch):
    """Redirect all on-disk state into a temp folder for each test."""
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(server, "BRIEFS_DIR", tmp_path / "briefs")


# --- helpers ----------------------------------------------------------------

class FakePaper:
    """Mimics the attributes search_arxiv reads off an arxiv result."""

    def __init__(self, idx: int):
        self._id = f"2606.{idx:05d}v1"
        self.title = f"Paper number {idx}"
        self.authors = [type("A", (), {"name": f"Author {idx}"})()]
        self.published = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
        self.summary = f"Summary of paper {idx}."
        self.entry_id = f"http://arxiv.org/abs/{self._id}"

    def get_short_id(self) -> str:
        return self._id


def _fake_results(count: int):
    papers = [FakePaper(i) for i in range(count)]
    return lambda search: iter(papers)


# --- _slug ------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("RAG evaluation!", "rag-evaluation"),
        ("  Multi-Agent   Systems  ", "multi-agent-systems"),
        ("", "topic"),
        ("!!!", "topic"),
    ],
)
def test_slug(raw, expected):
    assert server._slug(raw) == expected


def test_slug_is_length_capped():
    assert len(server._slug("x" * 200)) <= 60


# --- state load/save --------------------------------------------------------

def test_state_roundtrip():
    assert server._load_state("topic a")["seen_ids"] == []  # default when missing
    server._save_state("topic a", {"topic": "topic a", "seen_ids": ["1"], "briefs": []})
    assert server._load_state("topic a")["seen_ids"] == ["1"]


def test_corrupt_state_falls_back(tmp_path):
    server.STATE_DIR.mkdir(parents=True, exist_ok=True)
    server._state_path("broken").write_text("{not valid json", encoding="utf-8")
    # Should not raise - returns the safe default.
    assert server._load_state("broken")["seen_ids"] == []


# --- record_brief / get_seen_paper_ids --------------------------------------

def test_record_brief_saves_and_marks_seen():
    out = server.record_brief("rag", ["a", "b"], "# Brief\nhello")
    assert out["total_seen"] == 2
    assert server.BRIEFS_DIR.exists()
    assert list(server.BRIEFS_DIR.glob("*.md"))  # a file was written
    assert set(server.get_seen_paper_ids("rag")) == {"a", "b"}


def test_record_brief_dedupes_across_runs():
    server.record_brief("rag", ["a", "b"], "first")
    server.record_brief("rag", ["b", "c"], "second")
    # 'b' must not be duplicated; result is the sorted union.
    assert server.get_seen_paper_ids("rag") == ["a", "b", "c"]
    assert len(server.list_past_briefs("rag")) == 2


# --- search_arxiv (mocked network) ------------------------------------------

def test_search_arxiv_returns_shaped_dicts(monkeypatch):
    monkeypatch.setattr(server._arxiv_client, "results", _fake_results(5))
    results = server.search_arxiv("anything", max_results=3)
    assert len(results) == 3
    first = results[0]
    assert set(first) == {"id", "title", "authors", "published", "summary", "url"}
    assert first["authors"] == ["Author 0"]


def test_search_arxiv_clamps_max_results(monkeypatch):
    monkeypatch.setattr(server._arxiv_client, "results", _fake_results(50))
    results = server.search_arxiv("anything", max_results=999)  # cap is 25
    assert len(results) == server.MAX_RESULTS_CAP


def test_search_arxiv_empty_topic_returns_empty():
    assert server.search_arxiv("   ") == []
