from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.api.deps import get_email_service, require_role
from app.database.session import get_db
from app.models.approval_request import ApprovalRequest, ApprovalStatus
from app.models.ticket import Ticket, TicketPriority
from app.models.user import User, UserRole
from app.services.email_service import EmailService

router = APIRouter(prefix="/approvals", tags=["approvals"])

# Bound once so it's a single, named, importable/overridable dependency -
# calling require_role(...) inline in each route decorator would produce a
# fresh unnamed closure per route, which tests can't target.
require_manager = require_role(UserRole.MANAGER)


class ApprovalSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticket_id: int
    requester_id: int
    requested_priority: TicketPriority
    reason: str
    requested_at: datetime


class ApprovalDecisionRequest(BaseModel):
    approved: bool


class ApprovalDecisionResponse(BaseModel):
    status: ApprovalStatus


@router.get("", response_model=list[ApprovalSummary])
def list_pending_approvals(
    current_user: User = Depends(require_manager),
    session: Session = Depends(get_db),
) -> list[ApprovalSummary]:
    approvals = (
        session.query(ApprovalRequest)
        .filter_by(status=ApprovalStatus.PENDING)
        .order_by(ApprovalRequest.requested_at)
        .all()
    )
    return [ApprovalSummary.model_validate(a) for a in approvals]


@router.post("/{approval_id}/decide", response_model=ApprovalDecisionResponse)
def decide_approval(
    approval_id: int,
    decision: ApprovalDecisionRequest,
    current_user: User = Depends(require_manager),
    session: Session = Depends(get_db),
    email_service: EmailService = Depends(get_email_service),
) -> ApprovalDecisionResponse:
    approval = session.get(ApprovalRequest, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval request not found")

    new_status = ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED

    # Atomic conditional update, not read-then-write: prevents two
    # concurrent decisions on the same request both succeeding. rowcount
    # of 0 here means the request existed (checked above) but was no
    # longer pending by the time this ran.
    result = session.execute(
        update(ApprovalRequest)
        .where(
            ApprovalRequest.id == approval_id, ApprovalRequest.status == ApprovalStatus.PENDING
        )
        .values(status=new_status, decided_by_id=current_user.id, decided_at=datetime.now(UTC))
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=409, detail="This request has already been decided")

    if decision.approved:
        # The only place in the whole app anything the LLM proposed
        # actually takes effect - gated by require_manager above.
        ticket = session.get(Ticket, approval.ticket_id)
        ticket.priority = approval.requested_priority

    session.commit()

    requester = session.get(User, approval.requester_id)
    email_service.send(
        to=requester.email,
        subject=f"Your escalation request for ticket #{approval.ticket_id} was {new_status.value}",
        body=(
            f"Your request to escalate ticket #{approval.ticket_id} to "
            f"{approval.requested_priority.value} priority was {new_status.value} "
            f"by {current_user.name}."
        ),
    )

    return ApprovalDecisionResponse(status=new_status)
