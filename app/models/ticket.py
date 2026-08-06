import enum
from datetime import datetime

from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base
from app.models._enum_utils import enum_values


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str]
    status: Mapped[TicketStatus] = mapped_column(
        SqlEnum(TicketStatus, values_callable=enum_values)
    )
    priority: Mapped[TicketPriority] = mapped_column(
        SqlEnum(TicketPriority, values_callable=enum_values)
    )
    created_at: Mapped[datetime]
    resolved_at: Mapped[datetime | None] = mapped_column(default=None)
