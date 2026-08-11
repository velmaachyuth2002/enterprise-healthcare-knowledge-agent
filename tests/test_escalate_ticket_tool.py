from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.models.approval_request import ApprovalRequest, ApprovalStatus
from app.models.ticket import Ticket, TicketPriority, TicketStatus
from app.models.user import User, UserRole
from app.tools.escalate_ticket_tool import EscalateTicketInput, EscalateTicketTool


def _add_ticket_and_requester(db_session: Session) -> tuple[Ticket, User]:
    ticket = Ticket(
        subject="Claims submission blocked",
        status=TicketStatus.OPEN,
        priority=TicketPriority.MEDIUM,
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


def test_creates_a_pending_approval_request_without_changing_the_ticket(
    db_session: Session,
) -> None:
    ticket, requester = _add_ticket_and_requester(db_session)
    tool = EscalateTicketTool(db_session)

    result = tool.run(
        EscalateTicketInput(
            ticket_id=ticket.id,
            priority=TicketPriority.URGENT,
            reason="Affecting claims submission",
        ),
        requester_id=requester.id,
    )

    assert result.found is True
    assert result.approval_request_id is not None

    approval = db_session.query(ApprovalRequest).one()
    assert approval.ticket_id == ticket.id
    assert approval.requester_id == requester.id
    assert approval.requested_priority == TicketPriority.URGENT
    assert approval.reason == "Affecting claims submission"
    assert approval.status == ApprovalStatus.PENDING

    # The whole point of this tool: the ticket itself is untouched.
    db_session.refresh(ticket)
    assert ticket.priority == TicketPriority.MEDIUM


def test_returns_not_found_for_a_nonexistent_ticket_without_creating_a_request(
    db_session: Session,
) -> None:
    _, requester = _add_ticket_and_requester(db_session)
    tool = EscalateTicketTool(db_session)

    result = tool.run(
        EscalateTicketInput(ticket_id=999, priority=TicketPriority.URGENT, reason="x"),
        requester_id=requester.id,
    )

    assert result.found is False
    assert db_session.query(ApprovalRequest).count() == 0


def test_result_id_matches_the_created_approval_request(db_session: Session) -> None:
    ticket, requester = _add_ticket_and_requester(db_session)
    tool = EscalateTicketTool(db_session)

    result = tool.run(
        EscalateTicketInput(ticket_id=ticket.id, priority=TicketPriority.HIGH, reason="x"),
        requester_id=requester.id,
    )

    approval = db_session.query(ApprovalRequest).one()
    assert result.approval_request_id == approval.id
