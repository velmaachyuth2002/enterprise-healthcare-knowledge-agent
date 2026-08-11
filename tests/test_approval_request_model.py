from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.approval_request import ApprovalRequest, ApprovalStatus
from app.models.ticket import Ticket, TicketPriority, TicketStatus
from app.models.user import User, UserRole


def _add_ticket_and_requester(db_session: Session) -> tuple[Ticket, User]:
    ticket = Ticket(
        subject="Claims submission blocked",
        status=TicketStatus.OPEN,
        priority=TicketPriority.HIGH,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    requester = User(
        email="employee@medflow.example",
        name="Alex Chen",
        hashed_password="not-a-real-hash",
        role=UserRole.EMPLOYEE,
    )
    db_session.add_all([ticket, requester])
    db_session.commit()
    return ticket, requester


def test_approval_request_round_trips_through_the_database(db_session: Session) -> None:
    ticket, requester = _add_ticket_and_requester(db_session)

    db_session.add(
        ApprovalRequest(
            ticket_id=ticket.id,
            requester_id=requester.id,
            requested_priority=TicketPriority.URGENT,
            reason="Affecting claims submission",
            requested_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    approval = db_session.query(ApprovalRequest).one()

    assert approval.ticket_id == ticket.id
    assert approval.requester_id == requester.id
    assert approval.requested_priority == TicketPriority.URGENT
    assert approval.reason == "Affecting claims submission"
    assert approval.decided_by_id is None
    assert approval.decided_at is None


def test_status_defaults_to_pending(db_session: Session) -> None:
    ticket, requester = _add_ticket_and_requester(db_session)

    db_session.add(
        ApprovalRequest(
            ticket_id=ticket.id,
            requester_id=requester.id,
            requested_priority=TicketPriority.URGENT,
            reason="Affecting claims submission",
            requested_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    approval = db_session.query(ApprovalRequest).one()

    assert approval.status == ApprovalStatus.PENDING


def test_status_is_stored_as_its_lowercase_value_not_the_enum_name(db_session: Session) -> None:
    # Same regression class as Ticket/User: SQLAlchemy's default Enum
    # behavior persists the member name ("PENDING"), not its value
    # ("pending"), unless told otherwise.
    ticket, requester = _add_ticket_and_requester(db_session)

    db_session.add(
        ApprovalRequest(
            ticket_id=ticket.id,
            requester_id=requester.id,
            requested_priority=TicketPriority.URGENT,
            reason="Affecting claims submission",
            requested_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    raw_status = db_session.execute(text("SELECT status FROM approval_requests")).scalar_one()

    assert raw_status == "pending"
