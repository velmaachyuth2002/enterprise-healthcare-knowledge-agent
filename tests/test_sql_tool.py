from datetime import date, datetime

from sqlalchemy.orm import Session

from app.models.ticket import Ticket, TicketPriority, TicketStatus
from app.tools.sql_tool import TicketCountInput, TicketCountTool, UnresolvedTicketsTool


def _make_ticket(**overrides: object) -> Ticket:
    defaults: dict[str, object] = {
        "subject": "Test ticket",
        "status": TicketStatus.OPEN,
        "priority": TicketPriority.MEDIUM,
        "created_at": datetime(2026, 7, 15),
    }
    defaults.update(overrides)
    return Ticket(**defaults)  # type: ignore[arg-type]


def test_counts_only_tickets_within_the_range(db_session: Session) -> None:
    db_session.add_all(
        [
            _make_ticket(created_at=datetime(2026, 7, 1)),  # start boundary: included
            _make_ticket(created_at=datetime(2026, 7, 31)),  # inside range
            _make_ticket(created_at=datetime(2026, 8, 1)),  # end boundary: excluded
            _make_ticket(created_at=datetime(2026, 6, 30)),  # before range: excluded
        ]
    )
    db_session.commit()

    tool = TicketCountTool(db_session)
    result = tool.run(TicketCountInput(start=date(2026, 7, 1), end=date(2026, 8, 1)))

    assert result.count == 2


def test_count_is_zero_when_no_tickets_in_range(db_session: Session) -> None:
    tool = TicketCountTool(db_session)

    result = tool.run(TicketCountInput(start=date(2026, 1, 1), end=date(2026, 2, 1)))

    assert result.count == 0


def test_lists_only_open_and_in_progress_tickets(db_session: Session) -> None:
    db_session.add_all(
        [
            _make_ticket(subject="Open one", status=TicketStatus.OPEN),
            _make_ticket(subject="In progress one", status=TicketStatus.IN_PROGRESS),
            _make_ticket(subject="Resolved one", status=TicketStatus.RESOLVED),
            _make_ticket(subject="Closed one", status=TicketStatus.CLOSED),
        ]
    )
    db_session.commit()

    tool = UnresolvedTicketsTool(db_session)
    result = tool.run()

    subjects = {t.subject for t in result.tickets}
    assert subjects == {"Open one", "In progress one"}


def test_no_unresolved_tickets_returns_empty_list(db_session: Session) -> None:
    tool = UnresolvedTicketsTool(db_session)

    result = tool.run()

    assert result.tickets == []
