from fastapi.testclient import TestClient

from app.api.routes import get_agent_graph
from app.main import app


class _FakeGraph:
    def invoke(self, state: dict) -> dict:
        return {"answer": f"stub answer for: {state['question']}"}


def test_ask_with_policy_question_returns_answer() -> None:
    client = TestClient(app)

    response = client.post("/ask", json={"question": "What is our provider onboarding policy?"})

    assert response.status_code == 200
    assert "Provider Onboarding Policy" in response.json()["answer"]


def test_ask_rejects_empty_question() -> None:
    client = TestClient(app)

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_uses_injected_graph_dependency() -> None:
    # Proves the DI seam itself: the route depends on `get_agent_graph`, not
    # on a concrete graph, so tests can swap in a fake without touching the
    # real PolicyTool or LangGraph at all.
    app.dependency_overrides[get_agent_graph] = lambda: _FakeGraph()
    client = TestClient(app)

    try:
        response = client.post("/ask", json={"question": "anything"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "stub answer for: anything"
