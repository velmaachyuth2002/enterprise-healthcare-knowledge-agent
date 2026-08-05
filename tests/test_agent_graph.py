from datetime import date, datetime

from sqlalchemy.orm import Session

from app.graph.agent_graph import _last_month_range, build_graph
from app.models.ticket import Ticket, TicketPriority, TicketStatus
from app.tools.policy_tool import PolicyLookupInput, PolicyLookupResult, PolicyTool
from app.tools.sql_tool import (
    TicketCountInput,
    TicketCountResult,
    TicketCountTool,
    UnresolvedTicketsResult,
    UnresolvedTicketsTool,
)


class _RecordingFakePolicyTool:
    """Test double so routing behavior can be verified without depending on
    PolicyTool's real hardcoded content."""

    def __init__(self, result: PolicyLookupResult) -> None:
        self._result = result
        self.called_with: PolicyLookupInput | None = None

    def run(self, params: PolicyLookupInput) -> PolicyLookupResult:
        self.called_with = params
        return self._result

    def known_topics(self) -> list[str]:
        return []


class _RecordingFakeTicketCountTool:
    def __init__(self, result: TicketCountResult) -> None:
        self._result = result
        self.called_with: TicketCountInput | None = None

    def run(self, params: TicketCountInput) -> TicketCountResult:
        self.called_with = params
        return self._result


class _RecordingFakeUnresolvedTicketsTool:
    def __init__(self, result: UnresolvedTicketsResult) -> None:
        self._result = result
        self.called = False

    def run(self) -> UnresolvedTicketsResult:
        self.called = True
        return self._result


def test_policy_question_routes_to_tool_and_answers_with_content() -> None:
    fake_count_tool = _RecordingFakeTicketCountTool(TicketCountResult(count=0))
    fake_unresolved_tool = _RecordingFakeUnresolvedTicketsTool(
        UnresolvedTicketsResult(tickets=[])
    )
    graph = build_graph(PolicyTool(), fake_count_tool, fake_unresolved_tool)

    result = graph.invoke({"question": "What is our provider onboarding policy?"})

    assert "Provider Onboarding Policy" in result["answer"]
    assert result["tool_result"].found is True
    assert fake_count_tool.called_with is None
    assert fake_unresolved_tool.called is False


def test_unrecognized_question_never_calls_any_tool() -> None:
    fake_policy_tool = _RecordingFakePolicyTool(PolicyLookupResult(found=True))
    fake_count_tool = _RecordingFakeTicketCountTool(TicketCountResult(count=0))
    fake_unresolved_tool = _RecordingFakeUnresolvedTicketsTool(
        UnresolvedTicketsResult(tickets=[])
    )
    graph = build_graph(fake_policy_tool, fake_count_tool, fake_unresolved_tool)

    result = graph.invoke({"question": "What's the weather today?"})

    assert fake_policy_tool.called_with is None
    assert fake_count_tool.called_with is None
    assert fake_unresolved_tool.called is False
    assert result["answer"] == "I don't have a way to answer that yet."


def test_policy_question_with_unknown_topic_reports_not_found() -> None:
    fake_policy_tool = _RecordingFakePolicyTool(PolicyLookupResult(found=False))
    fake_count_tool = _RecordingFakeTicketCountTool(TicketCountResult(count=0))
    fake_unresolved_tool = _RecordingFakeUnresolvedTicketsTool(
        UnresolvedTicketsResult(tickets=[])
    )
    graph = build_graph(fake_policy_tool, fake_count_tool, fake_unresolved_tool)

    result = graph.invoke({"question": "What is our policy on time travel?"})

    assert fake_policy_tool.called_with is not None
    assert result["answer"] == "I couldn't find a policy on that topic."


def test_ticket_count_question_calls_only_the_ticket_count_tool() -> None:
    fake_policy_tool = _RecordingFakePolicyTool(PolicyLookupResult(found=True))
    fake_count_tool = _RecordingFakeTicketCountTool(TicketCountResult(count=3))
    fake_unresolved_tool = _RecordingFakeUnresolvedTicketsTool(
        UnresolvedTicketsResult(tickets=[])
    )
    graph = build_graph(
        fake_policy_tool,
        fake_count_tool,
        fake_unresolved_tool,
        today=lambda: date(2026, 8, 5),
    )

    result = graph.invoke({"question": "How many tickets were opened last month?"})

    assert fake_policy_tool.called_with is None
    assert fake_unresolved_tool.called is False
    assert fake_count_tool.called_with == TicketCountInput(
        start=date(2026, 7, 1), end=date(2026, 8, 1)
    )
    assert result["answer"] == "3 ticket(s) were opened in that period."


def test_unresolved_question_calls_only_the_unresolved_tickets_tool() -> None:
    fake_policy_tool = _RecordingFakePolicyTool(PolicyLookupResult(found=True))
    fake_count_tool = _RecordingFakeTicketCountTool(TicketCountResult(count=0))
    fake_unresolved_tool = _RecordingFakeUnresolvedTicketsTool(
        UnresolvedTicketsResult(tickets=[])
    )
    graph = build_graph(fake_policy_tool, fake_count_tool, fake_unresolved_tool)

    result = graph.invoke({"question": "Generate a report of unresolved tickets."})

    assert fake_policy_tool.called_with is None
    assert fake_count_tool.called_with is None
    assert fake_unresolved_tool.called is True
    assert result["answer"] == "There are no unresolved tickets."


def test_ticket_count_end_to_end_with_real_tool_and_database(db_session: Session) -> None:
    db_session.add_all(
        [
            Ticket(
                subject="In range",
                status=TicketStatus.OPEN,
                priority=TicketPriority.MEDIUM,
                created_at=datetime(2026, 7, 15),
            ),
            Ticket(
                subject="Out of range",
                status=TicketStatus.OPEN,
                priority=TicketPriority.MEDIUM,
                created_at=datetime(2026, 6, 1),
            ),
        ]
    )
    db_session.commit()

    graph = build_graph(
        PolicyTool(),
        TicketCountTool(db_session),
        UnresolvedTicketsTool(db_session),
        today=lambda: date(2026, 8, 5),
    )

    result = graph.invoke({"question": "How many tickets were opened last month?"})

    assert result["answer"] == "1 ticket(s) were opened in that period."


def test_unresolved_end_to_end_with_real_tool_and_database(db_session: Session) -> None:
    db_session.add_all(
        [
            Ticket(
                subject="Still open",
                status=TicketStatus.OPEN,
                priority=TicketPriority.HIGH,
                created_at=datetime(2026, 7, 1),
            ),
            Ticket(
                subject="Already resolved",
                status=TicketStatus.RESOLVED,
                priority=TicketPriority.LOW,
                created_at=datetime(2026, 7, 1),
            ),
        ]
    )
    db_session.commit()

    graph = build_graph(
        PolicyTool(), TicketCountTool(db_session), UnresolvedTicketsTool(db_session)
    )

    result = graph.invoke({"question": "Generate a report of unresolved tickets."})

    assert result["answer"] == "1 unresolved ticket(s): Still open"


def test_last_month_range_for_a_normal_month() -> None:
    start, end = _last_month_range(date(2026, 8, 5))

    assert start == date(2026, 7, 1)
    assert end == date(2026, 8, 1)


def test_last_month_range_rolls_back_to_december_of_previous_year() -> None:
    start, end = _last_month_range(date(2026, 1, 15))

    assert start == date(2025, 12, 1)
    assert end == date(2026, 1, 1)
