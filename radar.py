"""Research Radar — command-line interface.

Runs the Scout -> Analyst -> Briefer pipeline for a topic and streams each agent's
output. The finished brief is saved by the Briefer via the MCP server (data/briefs/).

Examples:
    python radar.py "retrieval augmented generation evaluation"
    python radar.py "diffusion models for audio" --retries 3

The exit code is 0 on success and 1 on failure, so it composes with scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import warnings

# --- Make output robust and quiet -------------------------------------------
# Windows consoles default to a legacy codepage that can't print Unicode (emoji,
# accented author names). Force UTF-8 so briefs render instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ADK/MCP are chatty at import/run time; keep the CLI output focused on the brief.
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from google.adk.runners import Runner  # noqa: E402  (after stdout/log setup)
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from research_radar.agent import MODEL, root_agent  # noqa: E402

APP = "research_radar"


def _explain_error(err: Exception) -> str:
    """Turn a raw exception into a short, actionable message for the user.

    These are the real failure modes we hit while building (see docs lesson 06):
    quota, transient overload, and a network/DNS block on the Gemini host.
    """
    msg = str(err)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return (
            "Gemini quota exceeded for this model today.\n"
            "  • Try again later, or set a different model in .env, e.g.:\n"
            "      GEMINI_MODEL=gemini-2.5-flash-lite\n"
            "  • Each run makes several model calls, so free tiers run out fast."
        )
    if "getaddrinfo" in msg or "ConnectError" in msg or "Failed to establish" in msg:
        return (
            "Couldn't reach the Gemini API (network/DNS).\n"
            "  • Some networks block 'generativelanguage.googleapis.com'.\n"
            "  • Try a mobile hotspot or a VPN, then re-run."
        )
    if "API key" in msg or "PERMISSION_DENIED" in msg or "API_KEY" in msg:
        return (
            "Gemini rejected the API key.\n"
            "  • Check GOOGLE_API_KEY in your .env (get one at "
            "https://aistudio.google.com/apikey)."
        )
    return f"Unexpected error: {msg}"


def _is_transient(err: Exception) -> bool:
    """503/overload errors are worth an automatic retry; quota/auth are not."""
    msg = str(err)
    return "UNAVAILABLE" in msg or "503" in msg or "DEADLINE_EXCEEDED" in msg


async def _run_once(topic: str) -> None:
    """Execute the full pipeline a single time, printing each agent's output."""
    sessions = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP, session_service=sessions)
    await sessions.create_session(app_name=APP, user_id="cli", session_id="cli")

    message = types.Content(role="user", parts=[types.Part(text=topic)])
    async for event in runner.run_async(
        user_id="cli", session_id="cli", new_message=message
    ):
        if event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts).strip()
            if text:
                print(f"\n── {event.author} ──\n{text}")


async def run(topic: str, retries: int = 2) -> int:
    """Run the pipeline, retrying on transient errors. Returns an exit code."""
    print(f"\n\U0001f52d Research Radar  ·  model: {MODEL}\n  topic: {topic}")
    print("=" * 64)

    attempt = 0
    while True:
        try:
            await _run_once(topic)
            print("\n" + "=" * 64)
            print("✅ Done. Brief saved under data/briefs/.")
            return 0
        except Exception as err:  # noqa: BLE001 — top-level CLI guard
            if _is_transient(err) and attempt < retries:
                attempt += 1
                wait = 5 * attempt
                print(f"\n⏳ Temporary error; retrying in {wait}s "
                      f"(attempt {attempt}/{retries})…")
                await asyncio.sleep(wait)
                continue
            print("\n" + "=" * 64)
            print("❌ " + _explain_error(err))
            return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a cited research brief for a topic from arXiv.",
    )
    parser.add_argument("topic", nargs="+", help="the research topic to brief")
    parser.add_argument(
        "--retries", type=int, default=2,
        help="automatic retries on transient (503) errors (default: 2)",
    )
    args = parser.parse_args()
    topic = " ".join(args.topic).strip()
    sys.exit(asyncio.run(run(topic, retries=args.retries)))


if __name__ == "__main__":
    main()
