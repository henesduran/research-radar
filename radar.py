"""Research Radar — headless CLI.

Runs the Scout -> Analyst -> Briefer pipeline once for a topic and prints each
agent's output as it goes. The finished brief is also saved by the Briefer via
the MCP server (see data/briefs/).

Usage:
    python radar.py "large language model agents"
"""

from __future__ import annotations

import asyncio
import sys

# Windows consoles default to a legacy codepage that can't print Unicode (emoji,
# accented author names). Force UTF-8 so briefs render instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from research_radar.agent import root_agent

APP = "research_radar"


async def run(topic: str) -> None:
    sessions = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP, session_service=sessions)
    await sessions.create_session(app_name=APP, user_id="cli", session_id="cli")

    message = types.Content(role="user", parts=[types.Part(text=topic)])

    print(f"\n🔭 Research Radar — topic: {topic}\n" + "=" * 60)
    async for event in runner.run_async(
        user_id="cli", session_id="cli", new_message=message
    ):
        if event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts).strip()
            if text:
                print(f"\n── {event.author} ──\n{text}")
    print("\n" + "=" * 60 + "\n✅ Done. Brief saved under data/briefs/.")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]).strip() or "large language model agents"
    asyncio.run(run(topic))
