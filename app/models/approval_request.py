import enum
from datetime import datetime

from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base
from app.models._enum_utils import enum_values
from app.models.ticket import TicketPriority


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"))
    requester_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    requested_priority: Mapped[TicketPriority] = mapped_column(
        SqlEnum(TicketPriority, values_callable=enum_values)
    )
    reason: Mapped[str]
    status: Mapped[ApprovalStatus] = mapped_column(
        SqlEnum(ApprovalStatus, values_callable=enum_values), default=ApprovalStatus.PENDING
    )
    requested_at: Mapped[datetime]
    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), default=None)
    decided_at: Mapped[datetime | None] = mapped_column(default=None)
