"""Shared, user-friendly error handling for the command-line tools.

Both radar.py (briefs) and ask.py (RAG questions) make Gemini calls and hit the same
failure modes, so the logic lives here once. `explain_error` turns a raw exception into
a short, actionable message; `is_transient` decides which errors are worth an automatic
retry (server hiccups) versus which are not (quota, bad key).
"""

from __future__ import annotations


def explain_error(err: Exception) -> str:
    """Turn a raw exception into a short, actionable message for the user."""
    msg = str(err)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return (
            "Gemini quota exceeded for this model today.\n"
            "  - Try again later, or set a different model in .env, e.g.:\n"
            "      GEMINI_MODEL=gemini-2.5-flash-lite\n"
            "  - Each run makes several model calls, so free tiers run out fast."
        )
    if "getaddrinfo" in msg or "ConnectError" in msg or "Failed to establish" in msg:
        return (
            "Couldn't reach the Gemini API (network/DNS).\n"
            "  - Some networks block 'generativelanguage.googleapis.com'.\n"
            "  - Try a mobile hotspot or a VPN, then re-run."
        )
    if "API key" in msg or "PERMISSION_DENIED" in msg or "API_KEY" in msg:
        return (
            "Gemini rejected the API key.\n"
            "  - Check GOOGLE_API_KEY in your .env (get one at "
            "https://aistudio.google.com/apikey)."
        )
    return f"Unexpected error: {msg}"


def is_transient(err: Exception) -> bool:
    """503/overload/timeout errors are worth an automatic retry; quota/auth are not."""
    msg = str(err)
    return "UNAVAILABLE" in msg or "503" in msg or "DEADLINE_EXCEEDED" in msg
