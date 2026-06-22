"""Pytest anchor.

Its presence at the project root makes pytest add the root to sys.path, so tests
can `import mcp_server.server` and `import research_radar` without extra config.
"""
