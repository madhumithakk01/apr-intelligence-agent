"""SQLite checkpointer for the orchestration graph -- CLAUDE.md section 15.

SQLite now, Postgres as the stated production path -- the same migration
point as the application database, and for the same reason: Render's
free tier does not persist the file across restarts.

Retention: a checkpoint row contains client application data verbatim,
so it falls under the same deletion trigger as the market cache and the
DB rows -- tied to bid outcome, not a generic TTL (CLAUDE.md section 2).
``purge_checkpoint_store`` is that trigger's hook; it deletes the whole
store rather than expiring rows by age, because "the bid concluded" is
the only event that authorizes deletion and it applies to everything at
once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DB = PROJECT_ROOT / "knowledge_db" / "orchestration_checkpoints.db"


def build_sqlite_checkpointer(db_path: Optional[Union[str, Path]] = None) -> SqliteSaver:
    """Long-lived saver over one connection.

    ``check_same_thread=False`` because LangGraph runs fanned-out
    branches on a worker thread pool -- several branches of the same run
    write checkpoints concurrently. SQLite serializes those writes
    itself; this only stops the driver from rejecting them for coming
    from a different thread than the one that opened the connection.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_CHECKPOINT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def build_in_memory_checkpointer() -> InMemorySaver:
    """For tests and dev runs. Resumability behaves identically to the
    SQLite saver within a single process; it simply keeps no file, so a
    test never leaves client-shaped data on disk."""
    return InMemorySaver()


def purge_checkpoint_store(db_path: Optional[Union[str, Path]] = None) -> bool:
    """Delete the checkpoint store outright. Call on the bid-outcome
    deletion trigger (CLAUDE.md section 2). Returns whether a file was
    actually removed."""
    path = Path(db_path) if db_path is not None else DEFAULT_CHECKPOINT_DB
    if not path.exists():
        return False
    path.unlink()
    return True
