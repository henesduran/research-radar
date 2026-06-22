"""Structural tests for the agent pipeline.

These check the wiring without making any model calls (construction only), so they
run offline and need no API key.
"""

from __future__ import annotations

from research_radar import agent


def test_pipeline_shape():
    root = agent.root_agent
    assert root.name == "research_radar"
    names = [a.name for a in root.sub_agents]
    assert names == ["Scout", "Analyst", "Briefer"]


def test_state_handoff_keys():
    # Each stage writes the key the next stage's prompt reads.
    by_name = {a.name: a for a in agent.root_agent.sub_agents}
    assert by_name["Scout"].output_key == "scout_findings"
    assert by_name["Analyst"].output_key == "analysis"
    assert by_name["Briefer"].output_key == "brief"


def test_least_privilege_tooling():
    by_name = {a.name: a for a in agent.root_agent.sub_agents}
    # The Analyst is a pure reasoner — it should have no tools.
    assert not by_name["Analyst"].tools
    # Scout and Briefer each get a toolset (search vs. write).
    assert by_name["Scout"].tools
    assert by_name["Briefer"].tools
