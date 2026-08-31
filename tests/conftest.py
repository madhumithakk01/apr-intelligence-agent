import sys
from pathlib import Path

import pytest

# The repo has no __init__.py anywhere and no pyproject.toml, so
# `uvicorn app.main:app` only works because uvicorn's import machinery
# adds the cwd to sys.path. Plain `pytest` doesn't do that for the repo
# root by default, so `import app...` needs this one-time shim.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _langsmith_tracing_off(monkeypatch):
    """Keep the suite hermetic even on a machine with LangSmith
    credentials exported: app.llm.tracing is a no-op without a key, so
    clearing these keeps every get_completion call from trying to reach
    LangSmith. A test that exercises tracing patches app.llm.tracing
    directly rather than relying on the environment."""
    for var in (
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(var, raising=False)
