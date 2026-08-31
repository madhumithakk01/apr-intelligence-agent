"""Deployment wiring -- CLAUDE.md sections 15, 16.

Structural guards, in the spirit of test_scoring_kernel.py's
source-reading checks: the Dockerfile, the CI workflows, the Render
blueprint, and the lint config must stay present and internally
coherent, so a change that quietly breaks the deploy fails here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# --- Dockerfile ---------------------------------------------------------


def test_dockerfile_matches_the_documented_shape():
    text = _read("Dockerfile")
    assert "FROM python:3.12-slim" in text  # CLAUDE.md section 15
    assert "requirements.txt" in text
    assert "COPY app ./app" in text
    assert "uvicorn app.main:app" in text
    assert "--port ${PORT:-8000}" in text  # honours the host-injected port


def test_dockerignore_keeps_local_state_and_tests_out_of_the_image():
    entries = {line.strip() for line in _read(".dockerignore").splitlines() if line.strip() and not line.startswith("#")}
    for needed in ("tests/", "knowledge_db/", "reports/", ".env", ".git", "data/"):
        assert needed in entries, f"{needed} missing from .dockerignore"


# --- CI workflows ----------------------------------------------------


def test_golden_subset_workflow_is_still_the_narrow_required_gate():
    wf = yaml.safe_load(_read(".github/workflows/golden-subset.yml"))
    (job,) = wf["jobs"].values()
    steps = " ".join(step.get("run", "") for step in job["steps"])
    assert "pytest tests/golden_subset" in steps
    assert "pytest tests/ -v" not in steps  # the full suite belongs to ci.yml, not this gate


def test_ci_workflow_adds_lint_full_suite_and_image_build():
    wf = yaml.safe_load(_read(".github/workflows/ci.yml"))
    jobs = wf["jobs"]
    assert set(jobs) == {"lint", "test", "docker-build"}

    lint_steps = " ".join(s.get("run", "") + " " + str(s.get("uses", "")) for s in jobs["lint"]["steps"])
    assert "ruff check ." in lint_steps

    test_steps = " ".join(s.get("run", "") for s in jobs["test"]["steps"])
    assert "pip install -r requirements.txt" in test_steps
    assert "pytest tests/" in test_steps

    build_uses = " ".join(str(s.get("uses", "")) for s in jobs["docker-build"]["steps"])
    assert "docker/build-push-action" in build_uses


def test_no_ci_workflow_configures_provider_secrets():
    # CLAUDE.md section 11: the suite is hermetic; a real key must never
    # be needed to pass CI.
    for name in ("ci.yml", "golden-subset.yml"):
        text = _read(f".github/workflows/{name}").lower()
        assert "secrets." not in text
        assert "api_key" not in text


# --- Render blueprint ---------------------------------------------


def test_render_blueprint_declares_one_free_docker_web_service():
    blueprint = yaml.safe_load(_read("render.yaml"))
    (service,) = blueprint["services"]
    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["plan"] == "free"
    assert service["dockerfilePath"] == "./Dockerfile"
    assert service["healthCheckPath"].startswith("/")


def test_render_blueprint_keeps_every_provider_key_unsynced():
    blueprint = yaml.safe_load(_read("render.yaml"))
    env_vars = {v["key"]: v for v in blueprint["services"][0]["envVars"]}
    for secret in ("GROQ_API_KEY", "TAVILY_API_KEY", "GEMINI_API_KEY", "LANGSMITH_API_KEY"):
        assert env_vars[secret].get("sync") is False
        assert "value" not in env_vars[secret]


# --- lint + dependency config -----------------------------------


def test_ruff_config_selects_pyflakes_and_targets_py312():
    config = tomllib.loads(_read("ruff.toml"))
    assert config["target-version"] == "py312"
    assert "F" in config["lint"]["select"]


def test_requirements_still_pin_langsmith_and_langgraph():
    reqs = _read("requirements.txt")
    assert "langsmith==" in reqs  # branch 17 regression guard
    assert "langgraph==" in reqs
