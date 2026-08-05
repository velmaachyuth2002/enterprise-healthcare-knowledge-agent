from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.ticket import Ticket, TicketPriority, TicketStatus


def test_ticket_round_trips_through_the_database(db_session: Session) -> None:
    db_session.add(
        Ticket(
            subject="Provider onboarding failing for new NPI numbers",
            status=TicketStatus.OPEN,
            priority=TicketPriority.HIGH,
            created_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    ticket = db_session.query(Ticket).one()

    assert ticket.subject == "Provider onboarding failing for new NPI numbers"
    assert ticket.status == TicketStatus.OPEN
    assert ticket.priority == TicketPriority.HIGH
    assert ticket.resolved_at is None


def test_status_is_stored_as_its_lowercase_value_not_the_enum_name(db_session: Session) -> None:
    # Guards against SQLAlchemy's default Enum behavior, which persists the
    # member *name* ("OPEN") rather than its value ("open") unless told
    # otherwise. A future SQL tool comparing against "open" would silently
    # match zero rows if this regressed.
    db_session.add(
        Ticket(
            subject="Test",
            status=TicketStatus.OPEN,
            priority=TicketPriority.LOW,
            created_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    raw_status = db_session.execute(text("SELECT status FROM tickets")).scalar_one()

    assert raw_status == "open"
