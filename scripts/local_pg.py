"""Docker-free Postgres 16 + pgvector for machines where Docker is unavailable.

Starts an embedded server (the `pgserver` dev dependency) in ./.pgdata, creates the `vector`
extension and writes its DSN into .env as DATABASE_URL. The server keeps running after this
script exits; stop it with `python scripts/local_pg.py stop`.

This is a convenience for development. The documented setup is `docker compose up`.
"""

import re
import subprocess
import sys
import time
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


def wait_for_slow_start(timeout: float = 120.0) -> str:
    """pgserver gives up after 10 s, but a cold start (crash recovery) can take longer.

    pg_ctl keeps starting the server in the background; read its port from postmaster.pid and
    wait until it accepts connections.
    """
    pid_file = DATA_DIR / "postmaster.pid"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = pid_file.read_text().splitlines() if pid_file.exists() else []
        if len(lines) >= 4 and lines[3].strip().isdigit():
            dsn = f"postgresql://postgres:@127.0.0.1:{lines[3].strip()}/postgres"
            try:
                psycopg.connect(dsn, connect_timeout=2).close()
                return dsn
            except psycopg.OperationalError:
                pass
        time.sleep(2)
    raise SystemExit("Postgres did not accept connections in time; see .pgdata/log")


def start() -> None:
    try:
        dsn = pgserver.get_server(DATA_DIR, cleanup_mode=None).get_uri()
    except subprocess.TimeoutExpired:  # slow cold start, not a failure
        dsn = wait_for_slow_start()
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
