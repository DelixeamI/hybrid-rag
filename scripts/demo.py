"""Start the API, ask three questions from the console, stop the API.

Needs Postgres with the corpus ingested (`make up ingest`). With LLM_MODE=recorded the questions
below are answered from cassettes (no network); other questions need a real LLM.
"""

import os
import subprocess
import sys
import time

import httpx

PORT = int(os.environ.get("DEMO_PORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"
QUESTIONS = [
    "Сколько календарных дней длится ежегодный основной оплачиваемый отпуск?",
    "можно ли уволить человека пока он на больничном",
    "Какой налог платит работодатель с заработной платы работника?",
]


def wait_until_healthy(server: subprocess.Popen[bytes], timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            sys.exit("API process exited before it became healthy")
        try:
            if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    sys.exit("API did not become healthy in time")


def show(question: str) -> None:
    body = httpx.post(f"{BASE}/ask", json={"question": question}, timeout=300).json()
    print(f"\n? {question}")
    print(f"  [{body['status']}] {body['answer']['answer']}")
    for citation in body["answer"]["citations"]:
        parts = f", части {', '.join(citation['parts'])}" if citation["parts"] else ""
        print(f"    - статья {citation['article']}{parts}: «{citation['quote']}»")
    print("  latency, ms:", {k: round(v) for k, v in body["timings_ms"].items()})


def main() -> None:
    command = [
        sys.executable, "-m", "uvicorn", "tkrag.api.service:app_from_env",
        "--factory", "--port", str(PORT), "--log-level", "warning",
    ]  # fmt: skip
    server = subprocess.Popen(command)
    try:
        wait_until_healthy(server)
        print("health:", httpx.get(f"{BASE}/health").json())
        for question in QUESTIONS:
            show(question)
    finally:
        server.terminate()
        server.wait(timeout=30)


if __name__ == "__main__":
    main()
