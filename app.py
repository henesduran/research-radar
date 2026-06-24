"""Research Radar - web UI (Streamlit).

A small browser front-end over the same agents the CLIs use:
- "Generate brief": runs the Scout -> Analyst -> Briefer pipeline for a topic.
- "Ask the papers": indexes the topic's papers (RAG) and answers a question with citations.

Run:  streamlit run app.py
(Needs GOOGLE_API_KEY in .env, same as the CLIs.)
"""

from __future__ import annotations

import asyncio

import streamlit as st
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from mcp_server import rag, server
from research_radar.agent import MODEL, researcher, root_agent
from research_radar.errors import explain_error

APP = "research_radar_web"


async def _collect(agent, prompt: str) -> list[tuple[str, str]]:
    """Run an agent once and return [(author, text), ...] for every spoken event."""
    sessions = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP, session_service=sessions)
    await sessions.create_session(app_name=APP, user_id="web", session_id="web")
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    chunks: list[tuple[str, str]] = []
    async for ev in runner.run_async(user_id="web", session_id="web", new_message=msg):
        if ev.content and ev.content.parts:
            text = "".join(p.text or "" for p in ev.content.parts).strip()
            if text:
                chunks.append((ev.author, text))
    return chunks


def _run(agent, prompt: str) -> list[tuple[str, str]]:
    return asyncio.run(_collect(agent, prompt))


st.set_page_config(page_title="Research Radar", page_icon="🔭", layout="wide")
st.title("🔭 Research Radar")
st.caption("A multi-agent research assistant for arXiv literature.")
st.sidebar.markdown(f"**Model:** `{MODEL}`")
st.sidebar.markdown(
    "Built with Google ADK, a custom MCP server, ChromaDB, and Gemini."
)

brief_tab, ask_tab = st.tabs(["Generate brief", "Ask the papers"])

with brief_tab:
    st.subheader("Topic in, cited brief out")
    topic = st.text_input(
        "Research topic",
        key="brief_topic",
        placeholder="retrieval augmented generation evaluation",
    )
    if st.button("Generate brief", type="primary", disabled=not topic):
        with st.spinner("Agents working: Scout -> Analyst -> Briefer..."):
            try:
                chunks = _run(root_agent, topic)
                brief = next(
                    (t for a, t in reversed(chunks) if a == "Briefer"),
                    chunks[-1][1] if chunks else "",
                )
                st.markdown(brief or "_No brief produced._")
            except Exception as err:  # noqa: BLE001
                st.error(explain_error(err))

with ask_tab:
    st.subheader("Ask questions, answered only from the papers (with citations)")
    col1, col2 = st.columns(2)
    topic2 = col1.text_input("Topic", key="ask_topic")
    question = col2.text_input("Question", key="ask_question")
    if st.button("Ask", type="primary", disabled=not (topic2 and question)):
        with st.spinner("Indexing papers (first time) and answering..."):
            try:
                if not rag._collection(topic2).count():
                    papers = server.search_arxiv(topic2, max_results=25)
                    rag.index_papers(topic2, papers)
                chunks = _run(researcher, f"Topic: {topic2}\nQuestion: {question}")
                answer = chunks[-1][1] if chunks else "_No answer produced._"
                st.markdown(answer)
            except Exception as err:  # noqa: BLE001
                st.error(explain_error(err))
