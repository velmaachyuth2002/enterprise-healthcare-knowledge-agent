from datetime import date, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.routes import get_agent_graph
from app.database.session import get_db
from app.graph.agent_graph import _last_month_range
from app.main import app
from app.models.ticket import Ticket, TicketPriority, TicketStatus


class _FakeGraph:
    def invoke(self, state: dict) -> dict:
        return {"answer": f"stub answer for: {state['question']}"}


def _override_get_db(session: Session):
    def _get_db():
        yield session

    return _get_db


def test_ask_with_policy_question_returns_answer(db_session: Session) -> None:
    # get_agent_graph now depends on get_db, which defaults to the real
    # sqlite:///./dev.db file - override it with the isolated in-memory
    # fixture session so this test never touches disk.
    app.dependency_overrides[get_db] = _override_get_db(db_session)
    client = TestClient(app)

    try:
        response = client.post(
            "/ask", json={"question": "What is our provider onboarding policy?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Provider Onboarding Policy" in response.json()["answer"]


def test_ask_rejects_empty_question() -> None:
    client = TestClient(app)

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_counts_tickets_end_to_end(db_session: Session) -> None:
    # Proves the full stack works for the new path: HTTP -> route -> real
    # graph -> real TicketCountTool -> real (in-memory) database. Seeds
    # using the app's own `_last_month_range` helper rather than hardcoding
    # a date, since the endpoint's planner uses the real clock (no `today`
    # injection at the API layer) - this way the test stays correct
    # regardless of which actual calendar month it runs in.
    start, _ = _last_month_range(date.today())
    db_session.add(
        Ticket(
            subject="Provider onboarding failing for new NPI numbers",
            status=TicketStatus.OPEN,
            priority=TicketPriority.HIGH,
            created_at=datetime.combine(start, datetime.min.time()),
        )
    )
    db_session.commit()

    app.dependency_overrides[get_db] = _override_get_db(db_session)
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
    # real tools, LangGraph, or the database at all.
    app.dependency_overrides[get_agent_graph] = lambda: _FakeGraph()
    client = TestClient(app)

    try:
        response = client.post("/ask", json={"question": "anything"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "stub answer for: anything"
