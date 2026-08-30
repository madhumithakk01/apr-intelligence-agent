import sys
from pathlib import Path

# The repo has no __init__.py anywhere and no pyproject.toml, so
# `uvicorn app.main:app` only works because uvicorn's import machinery
# adds the cwd to sys.path. Plain `pytest` doesn't do that for the repo
# root by default, so `import app...` needs this one-time shim.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
