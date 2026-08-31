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


@pytest.fixture(autouse=True)
def _isolated_shadow_ledger(tmp_path_factory, monkeypatch):
    """Point the shadow-mode engagement ledger
    (app.orchestration.shadow) at a throwaway file per test, so a run
    that reaches render_report never writes the real
    knowledge_db/shadow_ledger.json. A test that needs its own ledger
    constructs ShadowLedger directly."""
    from app.orchestration import shadow

    ledger_path = tmp_path_factory.mktemp("shadow") / "ledger.json"
    monkeypatch.setenv("SHADOW_LEDGER_PATH", str(ledger_path))
    shadow.reset_default_ledger()
    yield
    shadow.reset_default_ledger()
