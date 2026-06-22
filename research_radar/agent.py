"""Research Radar — ADK multi-agent pipeline.

Architecture (a SequentialAgent runs these three specialists in order, passing
results through shared session state):

    user topic
        |
        v
    [ Scout ]   -- MCP: search_arxiv, get_seen_paper_ids
        |  state["scout_findings"] (new papers as JSON)
        v
    [ Analyst ] -- pure reasoning: relevance scores, key contributions, themes
        |  state["analysis"]
        v
    [ Briefer ] -- MCP: record_brief  (writes the brief + marks papers seen)
        |  state["brief"]
        v
    Markdown research brief

The agents reach arXiv and the filesystem ONLY through the MCP server
(`mcp_server/server.py`). Each agent is given just the tools it needs
(`tool_filter`), which keeps the trust boundary tight.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioServerParameters

from . import prompts

# Load GOOGLE_API_KEY from the repo-root .env (copy .env.example -> .env first).
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# Gemini model. Default to 2.5-flash-lite: fast, capable enough for this task,
# and the most generous free tier of the current flash models — important since
# each pipeline run makes several model calls. gemini-2.5-flash is higher quality
# but has a much smaller free tier (~20 requests/day). Override with GEMINI_MODEL
# in .env (e.g. GEMINI_MODEL=gemini-2.5-flash for the highest-quality brief).
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Absolute path to the MCP server, launched with the SAME interpreter that runs
# the agent (so it uses this project's virtualenv).
_MCP_SERVER = str(_ROOT / "mcp_server" / "server.py")


def _toolset(tool_names: list[str]) -> McpToolset:
    """Connect to our MCP server over stdio, exposing only `tool_names`."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=[_MCP_SERVER],
            ),
            timeout=60.0,
        ),
        tool_filter=tool_names,
    )


scout = LlmAgent(
    name="Scout",
    model=MODEL,
    description="Finds new recent arXiv papers for a topic, skipping ones already briefed.",
    instruction=prompts.SCOUT_INSTRUCTION,
    tools=[_toolset(["search_arxiv", "get_seen_paper_ids"])],
    output_key="scout_findings",
)

analyst = LlmAgent(
    name="Analyst",
    model=MODEL,
    description="Scores relevance, extracts key contributions, and finds themes.",
    instruction=prompts.ANALYST_INSTRUCTION,
    output_key="analysis",
)

briefer = LlmAgent(
    name="Briefer",
    model=MODEL,
    description="Writes the final Markdown brief and saves it via the MCP server.",
    instruction=prompts.BRIEFER_INSTRUCTION,
    tools=[_toolset(["record_brief", "list_past_briefs"])],
    output_key="brief",
)

# The root agent ADK looks for. Runs Scout -> Analyst -> Briefer in sequence.
# Note: ADK 2.3 emits a forward-looking deprecation pointing to a future "Workflow"
# API that does not exist in this version yet, so SequentialAgent remains the correct
# choice here.
root_agent = SequentialAgent(
    name="research_radar",
    description="Topic in, cited research brief out: scouts arXiv, analyzes, and writes a brief.",
    sub_agents=[scout, analyst, briefer],
)
