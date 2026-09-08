import json
import logging

import pytest
from fastapi.testclient import TestClient

from tkrag.api.jsonlog import JsonFormatter
from tkrag.api.main import Service, create_app
from tkrag.generation.schemas import Answer, AskResult, Citation


def fake_ask(question: str) -> AskResult:
    if "налог" in question:
        return AskResult(
            status="refused",
            answer=Answer(answer="нет ответа", refused=True, reason="low_retrieval_score"),
            best_score=0.01,
            timings_ms={"retrieve": 3.0, "rerank": 4.0},
        )
    citation = Citation(chunk_id="115#0", article="115", quote="28 календарных дней")
    return AskResult(
        status="ok",
        answer=Answer(answer="28 календарных дней [1]", citations=[citation]),
        best_score=0.93,
        timings_ms={"retrieve": 3.0, "rerank": 4.0, "generate": 5.0, "validate": 0.1},
    )


@pytest.fixture
def client() -> TestClient:
    service = Service(ask=fake_ask, health=lambda: {"status": "ok", "chunks": 20})
    with TestClient(create_app(lambda: service)) as test_client:
        return test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_answer_with_citations(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Сколько дней отпуска?"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["answer"]["citations"][0] == {
        "chunk_id": "115#0",
        "article": "115",
        "parts": [],
        "quote": "28 календарных дней",
    }


def test_ask_refusal_is_a_normal_200(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Какой налог платить?"})
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "a"}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_logs_stage_latencies_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    service = Service(ask=fake_ask, health=lambda: {})
    with TestClient(create_app(lambda: service)) as client:
        client.post("/ask", json={"question": "Сколько дней отпуска?"})
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    record = next(r for r in lines if r["msg"] == "ask")
    assert record["latency_ms"] == {
        "retrieve": 3.0,
        "rerank": 4.0,
        "generate": 5.0,
        "validate": 0.1,
    }
    assert record["status"] == "ok"
    assert record["articles"] == ["115"]
    assert "request_id" in record


def test_json_formatter_emits_one_json_object_with_extras() -> None:
    record = logging.LogRecord("tkrag.api", logging.INFO, "f.py", 1, "ask", (), None)
    record.latency_ms = {"retrieve": 1.5}
    line = JsonFormatter().format(record)
    payload = json.loads(line)
    assert payload["msg"] == "ask"
    assert payload["latency_ms"] == {"retrieve": 1.5}
    assert "\n" not in line
