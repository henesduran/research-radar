# 🔭 Research Radar

![CI](https://github.com/henesduran/research-radar/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Built with Google ADK](https://img.shields.io/badge/built%20with-Google%20ADK-4285F4)
![Tools via MCP](https://img.shields.io/badge/tools-MCP-6E56CF)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Enter a research topic. Get back a cited, analyzed literature brief - and only the papers you haven't seen before.**

Research Radar is a multi-agent system built with [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/). It scouts arXiv for the most relevant recent work on a topic, evaluates and ranks each paper, synthesizes a structured Markdown brief, and remembers what it already showed you so every re-run surfaces *new* literature.

You can also **ask questions about the papers**: it embeds them into a local **vector database (ChromaDB)** and answers with citations (a full **RAG** pipeline). A small **web UI** wraps both the brief and the Q&A.

> Built for the Kaggle **AI Agents: Intensive Vibe Coding Capstone** (Freestyle track).

---

## The problem

Keeping up with a research area is a chore. arXiv posts hundreds of papers a day; a keyword search returns a firehose sorted by date, full of loosely-matching noise. Researchers, students, and engineers waste hours triaging titles to find the few papers that actually matter - and the next week they do it again, re-reading the same results.

A plain "summarize arXiv" script doesn't solve this. It can't judge relevance, it can't track what you've already seen, and it dumps a wall of text instead of a usable brief.

## The solution

Research Radar treats literature review as a **pipeline of specialized agents**, each doing one job well:

| Agent | Role | Tools |
|-------|------|-------|
| **Scout** | Finds the most relevant papers for the topic and filters out anything already covered in past briefs | `search_arxiv`, `get_seen_paper_ids` (MCP) |
| **Analyst** | Scores each paper's relevance (1-5), extracts its key contribution, and identifies cross-cutting themes | *(pure reasoning)* |
| **Briefer** | Writes the polished Markdown brief and persists it, marking its papers as "seen" | `record_brief`, `list_past_briefs` (MCP) |

The agents never touch arXiv or the filesystem directly - they go through a **custom MCP (Model Context Protocol) server**, which is the project's single trusted tool boundary. Each agent is granted *only* the tools it needs (least privilege).

The result: re-run the same topic next week and Scout silently skips everything you've already read, so the brief is always "what's new for *you*."

### Ask the papers (RAG)

Beyond the brief, a fourth agent answers questions about a topic's papers:

| Agent | Role | Tools |
|-------|------|-------|
| **Researcher** | Answers a question using only the retrieved papers, with inline citations; says "I don't know" if the papers don't cover it | `semantic_search` (MCP) |

It works as a classic **RAG** pipeline: the papers' abstracts are embedded with Gemini embeddings (`gemini-embedding-001`, using `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types) and stored in a persistent **ChromaDB** vector store; a question is embedded and the closest papers are retrieved and handed to the Researcher to answer with sources.

---

## Architecture

```mermaid
flowchart LR
    U([User: topic]) --> ROOT

    subgraph ROOT[SequentialAgent: research_radar]
        direction LR
        SCOUT[🔍 Scout] -->|state: scout_findings| ANALYST[🧠 Analyst]
        ANALYST -->|state: analysis| BRIEFER[✍️ Briefer]
    end

    BRIEFER --> OUT([📄 Markdown brief])

    SCOUT <-->|MCP / stdio| MCP[[MCP Server\nresearch-radar]]
    BRIEFER <-->|MCP / stdio| MCP
    MCP <-->|HTTP| ARXIV[(arXiv API)]
    MCP <-->|read/write| DISK[(data/: briefs + seen-state)]
```

Plain-text view:

```
user topic
    │
    ▼
┌─────────────────── SequentialAgent: research_radar ───────────────────┐
│  [Scout] ──state:scout_findings──▶ [Analyst] ──state:analysis──▶ [Briefer] │
│     ▲                                                                ▲   │
└─────┼────────────────────────────────────────────────────────────── ┼──┘
      │ MCP (stdio)                                          MCP (stdio)│
      ▼                                                                 ▼
            ┌─────────────── MCP Server (research-radar) ───────────────┐
            │  search_arxiv · get_seen_paper_ids · record_brief · ...    │
            └───────────┬───────────────────────────────┬──────────────┘
                        ▼                                ▼
                   arXiv API                    data/ (briefs + seen-state)
```

**Why this design**
- **Multi-agent over one mega-prompt:** separating gather / evaluate / synthesize keeps each agent's instructions focused and its output inspectable. State flows between stages via ADK `output_key`.
- **MCP server as the tool boundary:** the same server can be reused by any MCP-aware client, and scoping tools per agent (`tool_filter`) keeps the trust surface small.
- **Memory via the tool layer:** dedupe state lives behind `record_brief` / `get_seen_paper_ids`, so the "radar" behavior is a property of the tools, not buried in a prompt.

---

## Course concepts demonstrated

- ✅ **Agent / Multi-agent system (ADK)** - a `SequentialAgent` orchestrating three `LlmAgent` specialists with state hand-off, plus a standalone Researcher agent for Q&A.
- ✅ **MCP Server** - a custom `FastMCP` server (`mcp_server/server.py`) exposing six tools over stdio (arXiv search, dedupe/brief store, and RAG index/search), consumed by ADK via `McpToolset`.
- ✅ **RAG & embeddings** - papers are embedded with Gemini embeddings and stored in a persistent **ChromaDB** vector database; questions are answered by semantic retrieval plus cited generation (`mcp_server/rag.py`, `ask.py`).
- ✅ **Security features** - secrets in `.env` (git-ignored, never in code); per-agent least-privilege tool scoping; input caps in every MCP tool; agents have no arbitrary file or network access beyond the fixed tools.
- ✅ **Deployability** - containerized (`Dockerfile`) for Google Cloud Run (see Deployment).
- ➕ **Antigravity** - developed/iterated in Google's Antigravity IDE (shown in the demo video).

---

## Setup

**Requirements:** Python 3.10+ and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env       # then edit .env and paste your key
```

Your `.env`:
```
GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

## Usage

**CLI (one-shot brief):**
```bash
python radar.py "retrieval augmented generation evaluation"
python radar.py "diffusion models for audio" --retries 3   # auto-retry on transient errors
```
The brief prints to the console and is saved to `data/briefs/`. Run the same topic again later to get only newly-published / not-yet-seen papers. The CLI prints clear, actionable messages for the common failure modes (quota, network, bad key) and exits non-zero on failure so it composes with scripts.

**Interactive dev UI (ADK):**
```bash
adk web
```
Then open the printed URL, pick `research_radar`, and type a topic. The UI shows each agent and tool call in sequence - great for understanding the flow.

**Ask the papers (RAG):**
```bash
python ask.py "retrieval augmented generation evaluation" "What metrics evaluate RAG systems?"
```
On first use for a topic it indexes the papers (embeds their abstracts into ChromaDB), then answers your question with citations. Re-asking reuses the index; pass `--reindex` to refresh.

**Web UI:**
```bash
streamlit run app.py
```
Opens a browser app with two tabs: generate a brief, or ask the papers.

> **Model note:** the default model is `gemini-2.5-flash-lite` (fast, generous free tier). For the highest-quality brief, set `GEMINI_MODEL=gemini-2.5-flash` in `.env` (smaller free quota).

---

## Example output

A real brief produced by `python radar.py "graph neural networks for drug discovery"` is saved in [`examples/sample-brief.md`](examples/sample-brief.md). A snippet:

> **Executive summary** - This brief synthesizes recent advancements in applying Graph Neural Networks (GNNs) and related AI techniques to drug discovery ... with ongoing research focusing on novel architectures, efficiency, and robust evaluation.
>
> **Top Papers**: RAG-Enhanced Collaborative LLM Agents for Drug Discovery, Domain Knowledge Infused Conditional Generative Models, MECCH (Metapath Context Convolution HGNNs), Transformers are Graph Neural Networks, and more.

## Testing

Unit tests cover the MCP server's logic (input clamping, dedupe, state) and the
agent pipeline's wiring. They run fully offline - no network, no API key.

```bash
pip install -r requirements-dev.txt
pytest
```

## Deployment

The agent is container-ready. The `Dockerfile` serves the ADK agents over HTTP, so it
deploys cleanly to **Google Cloud Run** (or any container host):

```bash
# Build and run locally
docker build -t research-radar .
docker run -p 8080:8080 -e GOOGLE_API_KEY=your_key research-radar

# Deploy to Google Cloud Run (requires a GCP project + gcloud)
gcloud run deploy research-radar \
  --source . \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=FALSE \
  --set-secrets GOOGLE_API_KEY=GEMINI_KEY:latest \
  --allow-unauthenticated
```

Live deployment is optional - the project runs fully from the CLI; the container just
makes it portable. **Never bake your API key into the image** - pass it as an environment
variable or secret at run time.

## Project structure

```
.
├── research_radar/          # ADK agent package
│   ├── agent.py             # root_agent (Scout → Analyst → Briefer) + Researcher
│   ├── prompts.py           # per-agent instructions
│   ├── errors.py            # shared, user-friendly error handling
│   └── __init__.py
├── mcp_server/
│   ├── server.py            # FastMCP server: arXiv search + brief/seen store + RAG tools
│   └── rag.py               # ChromaDB vector store + Gemini embeddings
├── tests/                   # offline unit tests (pytest)
│   ├── test_server.py
│   ├── test_agent.py
│   └── test_rag.py
├── examples/
│   └── sample-brief.md      # a real generated brief
├── .github/workflows/ci.yml # runs the tests on every push (GitHub Actions)
├── radar.py                 # CLI: generate a brief
├── ask.py                   # CLI: ask the papers (RAG)
├── app.py                   # Streamlit web UI (brief + ask)
├── Dockerfile               # container image (Cloud Run ready)
├── data/                    # generated briefs + dedupe state (git-ignored)
├── requirements.txt         # runtime dependencies
├── requirements-dev.txt     # + test dependencies
├── pyproject.toml           # pytest configuration
├── LICENSE                  # MIT
└── .env.example
```

## Security notes

- **No secrets in code.** The API key is read from `.env`, which is git-ignored.
- **Least privilege.** Scout can search but not write; Briefer can write but not search - enforced with MCP `tool_filter`.
- **Bounded tools.** Every MCP tool clamps its inputs (result counts, topic length, stored-brief size) so a malformed agent request can't run away.
- **No arbitrary file/network access from agents.** All I/O is mediated by the MCP server's fixed tool set.

## Limitations & future work

- Source is arXiv only; adding more MCP tools (Semantic Scholar, conference proceedings, RSS) would broaden coverage.
- Relevance scoring is LLM-judged; a hybrid with embedding similarity would be more robust.
- A scheduled run (e.g., Cloud Run + Cloud Scheduler) would turn this into a true daily-digest service.
