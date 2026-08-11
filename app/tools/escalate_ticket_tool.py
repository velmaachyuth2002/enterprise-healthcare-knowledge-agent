from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.models.approval_request import ApprovalRequest
from app.models.ticket import Ticket, TicketPriority

_VALID_PRIORITIES = {p.value for p in TicketPriority}


class EscalateTicketInput(BaseModel):
    ticket_id: int
    # Deliberately a plain string, not TicketPriority, in the schema shown
    # to the model: Groq validates tool-call arguments against the JSON
    # schema server-side, before the response ever reaches our code - an
    # `enum: ["low", ...]` constraint gets enforced case-sensitively there,
    # and the model doesn't reliably match case (observed: "Urgent" vs
    # "urgent"), which fails the whole request with no chance for us to
    # normalize it. Validating case-insensitively ourselves, after the
    # fact, is the only point this can actually be fixed.
    priority: str = Field(description="One of: low, medium, high, urgent (any case).")
    reason: str

    @field_validator("priority")
    @classmethod
    def _normalize_priority(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(_VALID_PRIORITIES)}")
        return normalized


class EscalateTicketResult(BaseModel):
    found: bool
    ticket_id: int | None = None
    requested_priority: TicketPriority | None = None
    approval_request_id: int | None = None


class EscalateTicketTool:
    name = "escalate_ticket"
    description = (
        "Propose escalating a ticket to a higher priority. This does not change "
        "the ticket - it creates a request that a manager must approve first. "
        "Valid priority values: low, medium, high, urgent."
    )

    def __init__(self, session: Session) -> None:
        self._session = session

    def run(self, params: EscalateTicketInput, *, requester_id: int) -> EscalateTicketResult:
        ticket = self._session.get(Ticket, params.ticket_id)
        if ticket is None:
            return EscalateTicketResult(found=False)

        priority = TicketPriority(params.priority)
        approval_request = ApprovalRequest(
            ticket_id=ticket.id,
            requester_id=requester_id,
            requested_priority=priority,
            reason=params.reason,
            requested_at=datetime.now(UTC),
        )
        self._session.add(approval_request)
        self._session.commit()

        return EscalateTicketResult(
            found=True,
            ticket_id=ticket.id,
            requested_priority=priority,
            approval_request_id=approval_request.id,
        )
