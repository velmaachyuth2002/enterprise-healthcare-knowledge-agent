from datetime import UTC, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database.session import Base
from app.models.ticket import Ticket, TicketPriority, TicketStatus


def _in_memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_ticket_round_trips_through_the_database() -> None:
    session = _in_memory_session()

    session.add(
        Ticket(
            subject="Provider onboarding failing for new NPI numbers",
            status=TicketStatus.OPEN,
            priority=TicketPriority.HIGH,
            created_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
    )
    session.commit()

    ticket = session.query(Ticket).one()

    assert ticket.subject == "Provider onboarding failing for new NPI numbers"
    assert ticket.status == TicketStatus.OPEN
    assert ticket.priority == TicketPriority.HIGH
    assert ticket.resolved_at is None


def test_status_is_stored_as_its_lowercase_value_not_the_enum_name() -> None:
    # Guards against SQLAlchemy's default Enum behavior, which persists the
    # member *name* ("OPEN") rather than its value ("open") unless told
    # otherwise. A future SQL tool comparing against "open" would silently
    # match zero rows if this regressed.
    session = _in_memory_session()
    session.add(
        Ticket(
            subject="Test",
            status=TicketStatus.OPEN,
            priority=TicketPriority.LOW,
            created_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
    )
    session.commit()

    raw_status = session.execute(text("SELECT status FROM tickets")).scalar_one()

    assert raw_status == "open"
