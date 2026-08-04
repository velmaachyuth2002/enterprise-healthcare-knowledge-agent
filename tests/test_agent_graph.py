from app.graph.agent_graph import build_graph
from app.tools.policy_tool import PolicyLookupInput, PolicyLookupResult, PolicyTool


class _RecordingFakeTool:
    """Test double so routing behavior can be verified without depending on
    PolicyTool's real hardcoded content."""

    def __init__(self, result: PolicyLookupResult) -> None:
        self._result = result
        self.called_with: PolicyLookupInput | None = None

    def run(self, params: PolicyLookupInput) -> PolicyLookupResult:
        self.called_with = params
        return self._result

    def known_topics(self) -> list[str]:
        # Empty on purpose: these tests exercise the "policy"-keyword and
        # "no match at all" branches, neither of which needs a real topic list.
        return []


def test_policy_question_routes_to_tool_and_answers_with_content() -> None:
    graph = build_graph(PolicyTool())

    result = graph.invoke({"question": "What is our provider onboarding policy?"})

    assert "Provider Onboarding Policy" in result["answer"]
    assert result["tool_result"].found is True


def test_non_policy_question_never_calls_the_tool() -> None:
    fake_tool = _RecordingFakeTool(PolicyLookupResult(found=True))
    graph = build_graph(fake_tool)

    result = graph.invoke({"question": "What's the weather today?"})

    assert fake_tool.called_with is None
    assert result["answer"] == "I don't have a way to answer that yet."


def test_policy_question_with_unknown_topic_reports_not_found() -> None:
    fake_tool = _RecordingFakeTool(PolicyLookupResult(found=False))
    graph = build_graph(fake_tool)

    result = graph.invoke({"question": "What is our policy on time travel?"})

    assert fake_tool.called_with is not None
    assert result["answer"] == "I couldn't find a policy on that topic."
