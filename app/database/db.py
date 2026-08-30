from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "knowledge_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_DIR / 'apr.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


def migrate_schema() -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    statements = []

    if "market_products" in existing_tables:
        columns = {column["name"] for column in inspector.get_columns("market_products")}
        if "domain_key" not in columns:
            statements.append("ALTER TABLE market_products ADD COLUMN domain_key VARCHAR")
        if "vendor" not in columns:
            statements.append("ALTER TABLE market_products ADD COLUMN vendor VARCHAR")
        if "structured_json" not in columns:
            statements.append("ALTER TABLE market_products ADD COLUMN structured_json TEXT")

    if "applications" in existing_tables:
        columns = {column["name"] for column in inspector.get_columns("applications")}
        if "numeric_field_notes" not in columns:
            statements.append("ALTER TABLE applications ADD COLUMN numeric_field_notes TEXT")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
