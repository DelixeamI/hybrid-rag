"""Docker-free Postgres 16 + pgvector for machines where Docker is unavailable.

Starts an embedded server (the `pgserver` dev dependency) in ./.pgdata, creates the `vector`
extension and writes its DSN into .env as DATABASE_URL. The server keeps running after this
script exits; stop it with `python scripts/local_pg.py stop`.

This is a convenience for development. The documented setup is `docker compose up`.
"""

import re
import sys
from pathlib import Path

import pgserver
import psycopg

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / ".pgdata"
ENV_FILE = ROOT / ".env"


def write_dsn(dsn: str) -> None:
    text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    line = f"DATABASE_URL={dsn}"
    if re.search(r"^DATABASE_URL=.*$", text, flags=re.MULTILINE):
        text = re.sub(r"^DATABASE_URL=.*$", line, text, flags=re.MULTILINE)
    else:
        text += f"\n{line}\n"
    ENV_FILE.write_text(text, encoding="utf-8")


def start() -> None:
    server = pgserver.get_server(DATA_DIR, cleanup_mode=None)
    dsn = server.get_uri()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    write_dsn(dsn)
    print(f"local Postgres is up: {dsn}\nDATABASE_URL written to {ENV_FILE}")


def stop() -> None:
    server = pgserver.get_server(DATA_DIR, cleanup_mode="stop")
    server.cleanup()
    print("local Postgres stopped")


if __name__ == "__main__":
    stop() if sys.argv[1:] == ["stop"] else start()
