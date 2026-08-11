from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.approval_request import ApprovalRequest
from app.models.ticket import Ticket, TicketPriority


class EscalateTicketInput(BaseModel):
    ticket_id: int
    priority: TicketPriority
    reason: str


class EscalateTicketResult(BaseModel):
    found: bool
    ticket_id: int | None = None
    requested_priority: TicketPriority | None = None
    approval_request_id: int | None = None


class EscalateTicketTool:
    name = "escalate_ticket"
    description = (
        "Propose escalating a ticket to a higher priority. This does not change "
        "the ticket - it creates a request that a manager must approve first."
    )

    def __init__(self, session: Session) -> None:
        self._session = session

    def run(self, params: EscalateTicketInput, *, requester_id: int) -> EscalateTicketResult:
        ticket = self._session.get(Ticket, params.ticket_id)
        if ticket is None:
            return EscalateTicketResult(found=False)

        approval_request = ApprovalRequest(
            ticket_id=ticket.id,
            requester_id=requester_id,
            requested_priority=params.priority,
            reason=params.reason,
            requested_at=datetime.now(UTC),
        )
        self._session.add(approval_request)
        self._session.commit()

        return EscalateTicketResult(
            found=True,
            ticket_id=ticket.id,
            requested_priority=params.priority,
            approval_request_id=approval_request.id,
        )
