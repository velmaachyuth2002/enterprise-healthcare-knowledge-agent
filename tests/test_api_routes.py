import json
from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.routes import get_agent_graph, get_groq_client
from app.database.session import get_db
from app.main import app
from app.models.ticket import Ticket, TicketPriority, TicketStatus


class _FakeGraph:
    def invoke(self, state: dict) -> dict:
        return {"answer": f"stub answer for: {state['question']}"}


def _override_get_db(session: Session):
    def _get_db():
        yield session

    return _get_db


def _fake_groq_client_calling(tool_name: str, arguments: dict):
    """A fake Groq client whose single response is a tool call for
    `tool_name` with `arguments` - lets API tests exercise the real graph
    and real tools without hitting the network or depending on real model
    behavior. (Model behavior itself is proven separately, by the live
    tests in test_agent_graph.py and test_llm_gateway.py.)"""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments))
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    response = SimpleNamespace(choices=[choice], usage=usage)

    client = SimpleNamespace()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
    return client


def test_ask_with_policy_question_returns_answer(db_session: Session) -> None:
    app.dependency_overrides[get_db] = _override_get_db(db_session)
    app.dependency_overrides[get_groq_client] = lambda: _fake_groq_client_calling(
        "search_documents",
        {"query": "How long does provider onboarding take for a hospital or clinic?"},
    )
    client = TestClient(app)

    try:
        response = client.post(
            "/ask", json={"question": "What is our provider onboarding policy?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "provider_onboarding_guide.md" in response.json()["answer"]


def test_ask_rejects_empty_question() -> None:
    client = TestClient(app)

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_counts_tickets_end_to_end(db_session: Session) -> None:
    # Proves the full stack works: HTTP -> route -> real graph -> real
    # TicketCountTool -> real (in-memory) database, driven by a fake but
    # realistic Groq tool-call response.
    db_session.add(
        Ticket(
            subject="Provider onboarding failing for new NPI numbers",
            status=TicketStatus.OPEN,
            priority=TicketPriority.HIGH,
            created_at=datetime(2026, 7, 15),
        )
    )
    db_session.commit()

    app.dependency_overrides[get_db] = _override_get_db(db_session)
    app.dependency_overrides[get_groq_client] = lambda: _fake_groq_client_calling(
        "count_tickets_in_range", {"start": "2026-07-01", "end": "2026-08-01"}
    )
    client = TestClient(app)

    try:
        response = client.post(
            "/ask", json={"question": "How many tickets were opened last month?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "1 ticket(s) were opened in that period."


def test_ask_uses_injected_graph_dependency() -> None:
    # Proves the DI seam itself: the route depends on `get_agent_graph`, not
    # on a concrete graph, so tests can swap in a fake without touching the
    # real tools, LangGraph, Groq, or the database at all.
    app.dependency_overrides[get_agent_graph] = lambda: _FakeGraph()
    client = TestClient(app)

    try:
        response = client.post("/ask", json={"question": "anything"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "stub answer for: anything"
