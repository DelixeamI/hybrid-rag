"""Postgres connection and the tiny SQL migration runner."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def migrate(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply *.sql files in name order, once each. Returns names applied now."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY)")
    done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
    applied: list[str] = []
    for path in sorted(directory.glob("*.sql")):
        if path.name in done:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
        applied.append(path.name)
    conn.commit()
    return applied


# HNSW returns its `ef_search` nearest rows over the WHOLE table and only then applies the
# `document = ...` filter. With several documents in the table a small value starves the filtered
# query of candidates. 1000 is pgvector's maximum; beyond ~1000 chunks per table use iterative
# scans (pgvector >= 0.8) or one table per document.
HNSW_EF_SEARCH = 1000


def prepare_connection(conn: psycopg.Connection) -> None:
    """Per-connection setup: the vector type and a wide HNSW search."""
    register_vector(conn)
    conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    conn.commit()


def connect(dsn: str) -> psycopg.Connection:
    """Connect, make sure the schema exists, and prepare the connection."""
    conn = psycopg.connect(dsn)
    migrate(conn)
    prepare_connection(conn)
    return conn
