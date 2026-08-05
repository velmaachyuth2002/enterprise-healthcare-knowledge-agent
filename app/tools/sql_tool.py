from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.ticket import Ticket, TicketPriority, TicketStatus

_UNRESOLVED_STATUSES = (TicketStatus.OPEN, TicketStatus.IN_PROGRESS)


def _start_of_day(d: date) -> datetime:
    # `Ticket.created_at` is a naive DateTime column; converting explicitly
    # here (rather than comparing a date against a datetime column and
    # relying on implicit coercion) keeps the range boundary unambiguous
    # across dialects.
    return datetime.combine(d, time.min)


class TicketCountInput(BaseModel):
    start: date
    end: date


class TicketCountResult(BaseModel):
    count: int


class TicketCountTool:
    name = "count_tickets_in_range"
    description = "Count tickets created within a date range (inclusive start, exclusive end)."

    def __init__(self, session: Session) -> None:
        self._session = session

    def run(self, params: TicketCountInput) -> TicketCountResult:
        stmt = (
            select(func.count())
            .select_from(Ticket)
            .where(
                Ticket.created_at >= _start_of_day(params.start),
                Ticket.created_at < _start_of_day(params.end),
            )
        )
        count = self._session.execute(stmt).scalar_one()
        return TicketCountResult(count=count)


class TicketSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject: str
    status: TicketStatus
    priority: TicketPriority
    created_at: datetime


class UnresolvedTicketsResult(BaseModel):
    tickets: list[TicketSummary]


class UnresolvedTicketsTool:
    name = "list_unresolved_tickets"
    description = "List tickets that are still open or in progress."

    def __init__(self, session: Session) -> None:
        self._session = session

    def run(self) -> UnresolvedTicketsResult:
        stmt = select(Ticket).where(Ticket.status.in_(_UNRESOLVED_STATUSES))
        tickets = self._session.execute(stmt).scalars().all()
        return UnresolvedTicketsResult(tickets=[TicketSummary.model_validate(t) for t in tickets])
