"""
tests/conftest.py
=================
Shared pytest fixtures and configuration.
Fixes the pytest-xvfb crash and registers custom markers.
"""

import pytest


# ── Register custom markers (suppresses PytestUnknownMarkWarning) ─────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests with no I/O")
    config.addinivalue_line("markers", "graph: LangGraph workflow integration tests")
    config.addinivalue_line("markers", "slow: tests that make real network calls")


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def groq_available() -> bool:
    """True if GROQ_API_KEY is set in environment."""
    import os
    return bool(os.getenv("GROQ_API_KEY"))


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """True if local Ollama is reachable."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture
def echo_target():
    from arca.targets.connectors import EchoTarget
    return EchoTarget(vulnerable=False)


@pytest.fixture
def vulnerable_echo_target():
    from arca.targets.connectors import EchoTarget
    return EchoTarget(vulnerable=True)


@pytest.fixture
def safe_system_prompt():
    return "You are a helpful, harmless, and honest assistant."


@pytest.fixture
def minimal_state(echo_target, safe_system_prompt):
    from arca.graph.workflow import ATTACK_VECTORS
    return {
        "messages":             [],
        "target_system_prompt": safe_system_prompt,
        "target_callable":      echo_target,
        "attack_records":       [],
        "current_vector":       None,
        "vectors_used":         [],
        "attack_budget":        len(ATTACK_VECTORS),
        "defender_mitigations": [],
        "report":               None,
        "phase":                "attack",
        "session_id":           "test-session-001",
    }